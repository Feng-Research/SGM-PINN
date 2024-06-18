from sympy import Symbol, Eq, Abs
import torch
import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_2d import Rectangle

from modulus.models.fully_connected import FullyConnectedArch
from modulus.domain.constraint import Constraint
from modulus.key import Key
from modulus.graph import Graph

from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.monitor import PointwiseMonitor
from modulus.domain.validator import PointwiseValidator
from modulus.domain.inferencer import PointwiseInferencer
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.turbulence_zero_eq import ZeroEquation
from modulus.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.key import Key


@modulus.main(config_path="conf_zeroEq", config_name="config_2000")
def run(cfg: ModulusConfig) -> None:

    # add constraints to solver
    # make geometry
    height = 0.1
    width = 0.1
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make list of nodes to unroll graph on
    ze = ZeroEquation(nu=1e-4, dim=2, time=False, max_distance=height / 2)
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ns.make_nodes()
        + ze.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y")],
        req_names=[
            Key("u", derivatives=[Key("x")]),
            Key("u", derivatives=[Key("y")]),
            Key("v", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y")]),
        ],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, device=device, requires_grad=True)
        )
        importance = (
            outvar["u__x"] ** 2
            + outvar["u__y"] ** 2
            + outvar["v__x"] ** 2
            + outvar["v__y"] ** 2
        ) ** 0.5 + 10
        return importance.cpu().detach().numpy()

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 1.5, "v": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(y, height / 2),
        importance_measure=importance_measure,
    )
    ldc_domain.add_constraint(top_wall, "top_wall")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=y < height / 2,
        importance_measure=importance_measure,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
        importance_measure=importance_measure,
    )
    ldc_domain.add_constraint(interior, "interior")

    # add validator
    mapping = {
        "Points:0": "x",
        "Points:1": "y",
        "U:0": "u",
        "U:1": "v",
        "p": "p",
        "d": "sdf",
        "nuT": "nu",
    }
    openfoam_var = csv_to_dict(
        to_absolute_path("openfoam/cavity_uniformVel_zeroEqn_refined.csv"), mapping
    )
    openfoam_var["x"] += -width / 2  # center OpenFoam data
    openfoam_var["y"] += -height / 2  # center OpenFoam data
    openfoam_var["nu"] += 1e-4  # effective viscosity
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y", "sdf"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "nu"]
    }
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        true_outvar=openfoam_outvar_numpy,
        batch_size=1024,
        plotter=ValidatorPlotter(),
        requires_grad=True,
    )
    ldc_domain.add_validator(openfoam_validator)

    # add inferencer data
    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=openfoam_invar_numpy,
        output_names=["u", "v", "p", "nu"],
        batch_size=1024,
        plotter=InferencerPlotter(),
        requires_grad=True,
    )
    ldc_domain.add_inferencer(grid_inference, "inf_data")

    # add monitors
    global_monitor = PointwiseMonitor(
        rec.sample_interior(4000),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["continuity"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]
                * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    ldc_domain.add_monitor(global_monitor)

    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
