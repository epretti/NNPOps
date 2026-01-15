[![GH Actions Status](https://github.com/openmm/nnpops/workflows/CI/badge.svg)](https://github.com/openmm/nnpops/actions?query=branch%3Amaster+workflow%3ACI)
[![Conda](https://img.shields.io/conda/v/conda-forge/nnpops.svg)](https://anaconda.org/conda-forge/nnpops)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/nnpops/badges/downloads.svg)](https://anaconda.org/conda-forge/nnpops)

# NNPOps

The goal of this project is to promote the use of neural network potentials (NNPs)
by providing highly optimized, open source implementations of bottleneck operations
that appear in popular potentials.  These are the core design principles.

1. Each operation is entirely self contained, consisting of only a few source files
that can easily be incorporated into any code that needs to use it.

2. Each operation has a simple, clearly documented API allowing it to be easily
used in any context.

3. We provide both CPU (pure C++) and CUDA implementations of all operations.

4. The CUDA implementations are highly optimized.  The CPU implementations are written
in a generally efficient way, but no particular efforts have been made to tune them
for optimum performance.

5. This code is designed for inference (running simulations), not training (creating
new potential functions).  It computes gradients with respect to particle positions,
not model parameters.

## Installation

### Install with Conda

A [conda](https://docs.conda.io/) package can be installed from the [conda-forge channel](https://anaconda.org/conda-forge/nnpops):
```bash
conda install -c conda-forge nnpops
```
If you don't have `conda`, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Build from source

#### Prerequisites

- *CUDA Toolkit* (https://developer.nvidia.com/cuda-downloads)
- *Miniconda* (https://docs.conda.io/en/latest/miniconda.html#linux-installers)

#### Build & install

- Get the source code
```bash
$ git clone https://github.com/openmm/NNPOps.git
```

- Set `CUDA_HOME`
```bash
$ export CUDA_HOME=/usr/local/cuda-11.2
```

- Crate and activate a *Conda* environment
```bash
$ cd NNPOps
$ conda env create -n nnpops -f environment.yml
$ conda activate nnpops
```

- Configure, build, and install
```bash
$ mkdir build && cd build
$ cmake .. \
        -DTorch_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')/Torch \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ make install
```

- Run the tests
```bash
$ ctest --verbose
```

## Operations

The following optimized operations are present in NNPOps and accessible from
Python using the listed classes or functions:

- ANI symmetry functions: `NNPOps.SymmetryFunctions.ANISymmetryFunctions`
- Continuous filter convolution (CFConv): `NNPOps.CFConv.CFConv`, `NNPOps.CFConv.CFConvNeighbors`
- Neighbor pair enumeration: `NNPOps.neighbors.getNeighborPairs()`
- Particle mesh Ewald (PME): `NNPOps.pme.PME`
