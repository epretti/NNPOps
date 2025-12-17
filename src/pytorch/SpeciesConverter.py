#
# Copyright (c) 2020-2021 Acellera
# Authors: Raimondas Galvelis
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

class TorchANISpeciesConverter(torch.nn.Module):

    def __init__(self, converter, atomic_nums: Tensor) -> None:

        super().__init__()

        # Convert atomic numbers to a list of species
        species = converter(atomic_nums)
        self.register_buffer('species', species)

        # TODO: not needed for PyTorch 2.9; do any supported versions require it?
        # self.conv_tensor = converter.conv_tensor # Just to make TorchScript happy :)

    def forward(self, atomic_nums: Tensor, nop: bool = False, _dont_use: bool = False) -> Tensor:
        # Match TorchANI signature and behavior exactly here
        if _dont_use:
            raise ValueError("_dont_use should never be set")
        if nop:
            return atomic_nums
        return self.species
