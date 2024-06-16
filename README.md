
# SGM-PINN
Code for SGM-PINN, DAC 2024 on Modulus 22.09
This repository contains **only** additional files and files that were edited from the original. This project includes source code provided by Nvidia.
This project also includes code from https://github.com/Feng-Research/SPADE/tree/main
and https://github.com/Feng-Research/HyperEF

## Usage
Interact within a docker environment based on the 22.09 container. Full requirements and install guide for the container is [here](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/getting_started/installation.html#install-the-docker-engine) with abbreviated instructions below. 

After following the installation below, from within the container you can navigate to `cd /sgm-examples/` and run any of the .py files within. 

To view the results use tensorboard from outside the container. E.g. `tensorboard --logdir=./sgm-examples/ldc --port=7007` and open a web browser at localhost:7007.

## Installation (Ubuntu 20.04 or later)
1. [Ensure the docker engine in installed.](https://docs.docker.com/engine/install/ubuntu/)
2.  Install the baseline image
  - ```sudo apt-get install nvidia-docker2```
  - `docker pull nvcr.io/nvidia/modulus/modulus:``22.09`
3. **Clone** this repository to a working directory of your choice.
4. Open a terminal **in the working directory.**
5. Create and enter a persistent container to install the remaining requirements.
```
    docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime nvidia -v ${PWD}/sgm-examples:/sgm-examples \
    -v ${PWD}/sgm-modulus:/sgm-modulus \
    -it nvcr.io/nvidia/modulus/modulus:22.09 bash
```
  - If this container is stopped or closed you can find it with `docker container ls -a`, then `start` or `stop` based on its NAME (tab to auto-complete). Then re-enter with `docker exec -w /sgm-examples/ -it <CONTAINER_ID> /bin/bash`.
6. Inside the container, install Julia and required packages
	- `curl -fsSL https://install.julialang.org | sh` and follow a standard install
	- Run `. /root/.bashrc` as indicated
	- Enter the Julia REPL, and press `]` to enter the pkg mode
	- Enter `add SparseArrays, LinearAlgebra, Clustering, NearestNeighbors, MAT, Distances, Metis, Arpack, Statistics, DelimitedFiles, StatsBase, Random, Debugger, Laplacians#master, LinearMaps, PyCall` and wait for install to complete
	- `Ctrl+C` and `Ctrl+D` to exit the REPL.
7. In the container, install the following via pip
	- `pip install pyjulia`
	- `pip install pycall`
	- `pip install julia`
	- `pip install hnswlib`
8. Copy the contents of sgm-modulus to /modulus/modulus `cp -r /sgm-modulus/* /modulus/modulus/` or run `. /sgm-modulus/load-SGM-PINN-SPADE.sh`

For convenience a script 'dEnter' is included to enter the most recently started container.
