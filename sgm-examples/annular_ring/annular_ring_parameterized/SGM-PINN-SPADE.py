from sympy import Symbol, Eq
import numpy as np
import torch

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry import Bounds, Parameterization, Parameter
from modulus.geometry.primitives_2d import Rectangle, Circle
from modulus.utils.sympy.functions import parabola
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.domain.inferencer import PointwiseInferencer
from modulus.domain.monitor import PointwiseMonitor
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.graph import Graph
from modulus.domain.constraint import Constraint
import numpy as np

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("r")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("sdf"),
                Key("area"), Key("r"),
                Key("u"), Key("v")],
        req_names=[
            Key("continuity"),
            Key("momentum_x"),
            Key("momentum_y"),
            Key("u", derivatives=[Key("x")]),
            Key("u", derivatives=[Key("y")]),
            Key("v", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y")]),
        ],
    ).to(device)

    def importance_measure(invar):
        ivars = [
               "continuity","momentum_x","momentum_y",
               "u__x","u__y",
               "v__x","v__y",
               ]
        outvar = importance_model_graph(
            Constraint._set_device(invar, device=device, requires_grad=True)
        )
        importance = np.concatenate([outvar[i].cpu().detach().numpy() for i in ivars],
                            axis = 1)
        return importance, ivars


    # add constraints to solver
    # specify params
    channel_length = (-6.732, 6.732)
    channel_width = (-1.0, 1.0)
    cylinder_center = (0.0, 0.0)
    outer_cylinder_radius = 2.0
    inner_cylinder_radius = Parameter("r")
    inner_cylinder_radius_ranges = (0.75, 1.1)
    inlet_vel = 1.5
    parameterization = Parameterization(
        {inner_cylinder_radius: inner_cylinder_radius_ranges}
    )

    # make geometry
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    outer_circle = Circle(cylinder_center, outer_cylinder_radius)
    inner_circle = Circle(
        (0, 0), inner_cylinder_radius, parameterization=parameterization
    )
    geo = (rec + outer_circle) - inner_circle

    # make annular ring domain
    domain = Domain()

    # inlet
    inlet_sympy = parabola(
        y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": inlet_sympy, "v": 0},
        batch_size=cfg.batch_size.inlet,
        batch_per_epoch=4000,
        criteria=Eq(x, channel_length[0]),
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        batch_per_epoch=4000,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
        batch_per_epoch=4000,
        criteria=(x > channel_length[0]) & (x < channel_length[1]),
    )
    domain.add_constraint(no_slip, "no_slip")

    KNN = 7
    coarse_level = 5
    warmup = 10000
    sample_ratio = .15
    batch_iterations = 7000
    iterations_rebuild = 60000
    local_grid_width = 3 
    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=1024,
        batch_per_epoch=8000,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
        importance_measure=importance_measure,
        mapping_function='default',
        KNN = KNN,
        coarse_level = coarse_level,
        warmup = warmup,
        sample_ratio = sample_ratio,
        sample_bounds = [15/100,70/100],
        batch_iterations = batch_iterations,
        initial_graph_vars = ["x","y","sdf",
            ],
        graph_vars = [
            "x", "y", 
            "sdf",
            #"u__r","v__r",
            #"continuity","momentum_x","momentum_y",
            ],
        SPADE_vars = [
            "x","y",
            "u__x","u__y",
            "v__x","v__y",
            ],
        LOSS_vars = ["continuity","momentum_x","momentum_y",
            #"u__r","u__r",
            #"v__r","v__r"
            ],
        local_grid_width = local_grid_width,
        iterations_rebuild = iterations_rebuild,
    )
    domain.add_constraint(interior, "interior")

    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_dot_vel": 2},
        batch_size=10,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    # r1
    openfoam_var_r1 = csv_to_dict(
        to_absolute_path("../openfoam/bend_finerInternal0.csv"), mapping
    )
    openfoam_var_r1["x"] += channel_length[0]  # center OpenFoam data
    openfoam_var_r1["r"] = np.zeros_like(openfoam_var_r1["x"]) + 1.0
    openfoam_invar_r1_numpy = {
        key: value for key, value in openfoam_var_r1.items() if key in ["x", "y", "r"]
    }
    openfoam_outvar_r1_numpy = {
        key: value for key, value in openfoam_var_r1.items() if key in ["u", "v", "p"]
    }
    # r875
    openfoam_var_r875 = csv_to_dict(
        to_absolute_path("../openfoam/annularRing_r_0.8750.csv"), mapping
    )
    openfoam_var_r875["x"] += channel_length[0]  # center OpenFoam data
    openfoam_var_r875["r"] = np.zeros_like(openfoam_var_r875["x"]) + 0.875
    openfoam_invar_r875_numpy = {
        key: value for key, value in openfoam_var_r875.items() if key in ["x", "y", "r"]
    }
    openfoam_outvar_r875_numpy = {
        key: value for key, value in openfoam_var_r875.items() if key in ["u", "v", "p"]
    }
    # r75
    openfoam_var_r75 = csv_to_dict(
        to_absolute_path("../openfoam/annularRing_r_0.750.csv"), mapping
    )
    openfoam_var_r75["x"] += channel_length[0]  # center OpenFoam data
    openfoam_var_r75["r"] = np.zeros_like(openfoam_var_r75["x"]) + 0.75
    openfoam_invar_r75_numpy = {
        key: value for key, value in openfoam_var_r75.items() if key in ["x", "y", "r"]
    }
    openfoam_outvar_r75_numpy = {
        key: value for key, value in openfoam_var_r75.items() if key in ["u", "v", "p"]
    }

    # r1
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_r1_numpy,
        true_outvar=openfoam_outvar_r1_numpy,
        batch_size=1024,
    )
    domain.add_validator(openfoam_validator)

    # r 875
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_r875_numpy,
        true_outvar=openfoam_outvar_r875_numpy,
        batch_size=1024,
    )
    domain.add_validator(openfoam_validator)

    # r 75
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=openfoam_invar_r75_numpy,
        true_outvar=openfoam_outvar_r75_numpy,
        batch_size=1024,
    )
    domain.add_validator(openfoam_validator)

    # add inferencer data
    for i, radius in enumerate(
        np.linspace(
            inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 10
        )
    ):
        radius = float(radius)
        sampled_interior = geo.sample_interior(
            1024,
            bounds=Bounds(
                {x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}
            ),
            parameterization={inner_cylinder_radius: radius},
        )
        point_cloud_inference = PointwiseInferencer(
            nodes=nodes,
            invar=sampled_interior,
            output_names=["u", "v", "p"],
            batch_size=1024,
        )
        domain.add_inferencer(point_cloud_inference, "inf_data" + str(i).zfill(5))

    # add monitors
    # metric for mass and momentum imbalance
    global_monitor = PointwiseMonitor(
        geo.sample_interior(
            1024,
            bounds=Bounds(
                {x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}
            ),
        ),
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
    domain.add_monitor(global_monitor)

    # metric for force on inner sphere
    for i, radius in enumerate(
        np.linspace(inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 3)
    ):
        radius = float(radius)
        force_monitor = PointwiseMonitor(
            inner_circle.sample_boundary(
                1024,
                parameterization={inner_cylinder_radius: radius},
            ),
            output_names=["p"],
            metrics={
                "force_x_r"
                + str(radius): lambda var: torch.sum(
                    var["normal_x"] * var["area"] * var["p"]
                ),
                "force_y_r"
                + str(radius): lambda var: torch.sum(
                    var["normal_y"] * var["area"] * var["p"]
                ),
            },
            nodes=nodes,
        )
        domain.add_monitor(force_monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
