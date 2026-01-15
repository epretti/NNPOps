#
# Copyright (c) 2020 Acellera, 2025 Stanford University and the Authors
# Authors: Raimondas Galvelis, Evan Pretti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import torch
from torch import Tensor

class ANISymmetryFunctions(torch.nn.Module):
    """
    PyTorch module for optimized ANI symmetry functions.
    """

    Holder = torch.classes.NNPOpsANISymmetryFunctions.Holder
    operation = torch.ops.NNPOpsANISymmetryFunctions.operation

    def __init__(self, numSpecies: int, Rcr: float, Rca: float,
                 EtaR: list[float], ShfR: list[float], EtaA: list[float],
                 Zeta: list[float], ShfA: list[float], ShfZ: list[float],
                 atomSpecies: list[int]) -> None:
        """
        Create an `ANISymmetryFunctions` instance.

        Parameters
        ----------
        numSpecies : int
            The number of species.
        Rcr : float
            The cutoff distance for the radial symmetry functions.
        Rca : float
            The cutoff distance for the angular symmetry functions.
        EtaR : list[float]
            The Gaussian scale parameters for the radial symmetry functions.
        ShfR : list[float]
            The Gaussian shift parameters for the radial symmetry functions.
        EtaA : list[float]
            The Gaussian scale parameters for the angular symmetry functions.
        Zeta : list[float]
            The exponents for the angular symmetry functions.
        ShfA : list[float]
            The Gaussian shift parameters for the angular symmetry functions.
        ShfZ : list[float]
            The shift angles for the angular symmetry functions.
        atomSpecies: list[int]
            The species indices for each of the atoms.
        """

        super().__init__()

        self.holder = ANISymmetryFunctions.Holder(numSpecies, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, atomSpecies)

    def forward(self, positions: Tensor, cell: Tensor | None = None) -> list[Tensor]:
        """
        Evaluate the ANI symmetry functions.

        Parameters
        ----------
        positions : Tensor
            Atomic positions.
        cell : Tensor, optional
            Box vectors for periodic boundary conditions, if provided.

        Returns
        -------
        [Tensor, Tensor]
            Values of the radial and angular symmetry functions for each atom.
        """

        return ANISymmetryFunctions.operation(self.holder, positions, cell)
