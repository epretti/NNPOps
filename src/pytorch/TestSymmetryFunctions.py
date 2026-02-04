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

import os
import pytest
import tempfile
import torch

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')

VALUE_TOL = 1e-5
GRADIENT_TOL = 1e-4

def test_import():
    import NNPOps
    import NNPOps.SymmetryFunctions

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99', 'water'])
def test_compare_with_reference(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.SymmetryFunctions import ANISymmetryFunctions

    device = torch.device(deviceString)

    test_case = torch.load(os.path.join(test_data_path, f'{molFile}.pt'))
    positions = test_case['positions'].to(device)
    positions.requires_grad = True
    cell = test_case['cell']
    if cell is not None:
        cell = cell.to(device)

    expected = test_case['output'].to(device)
    actual = torch.concat(ANISymmetryFunctions(**test_case['parameters'])(positions, cell), dim=1)
    total = torch.sum(actual)
    total.backward()

    assert torch.allclose(actual, expected, rtol=VALUE_TOL, atol=VALUE_TOL)
    assert torch.allclose(positions.grad, test_case['grad'].to(device), rtol=GRADIENT_TOL, atol=GRADIENT_TOL)

@pytest.mark.parametrize('deviceString', ['cpu', 'cuda'])
@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99', 'water'])
def test_model_serialization(deviceString, molFile):

    if deviceString == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.SymmetryFunctions import ANISymmetryFunctions

    device = torch.device(deviceString)

    test_case = torch.load(os.path.join(test_data_path, f'{molFile}.pt'))
    positions = test_case['positions'].to(device)
    positions.requires_grad = True
    cell = test_case['cell']
    if cell is not None:
        cell = cell.to(device)

    expected = test_case['output'].to(device)

    with tempfile.NamedTemporaryFile() as fd:

        torch.jit.script(ANISymmetryFunctions(**test_case['parameters'])).save(fd.name)
        actual = torch.concat(torch.jit.load(fd.name)(positions, cell), dim=1)

    total = torch.sum(actual)
    total.backward()

    assert torch.allclose(actual, expected, rtol=VALUE_TOL, atol=VALUE_TOL)
    assert torch.allclose(positions.grad, test_case['grad'].to(device), rtol=GRADIENT_TOL, atol=GRADIENT_TOL)

@pytest.mark.parametrize('molFile', ['1hvj', '1hvk', '2iuz', '3hkw', '3hky', '3lka', '3o99', 'water'])
def test_non_default_stream(molFile):

    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    from NNPOps.SymmetryFunctions import ANISymmetryFunctions

    device = torch.device('cuda')

    test_case = torch.load(os.path.join(test_data_path, f'{molFile}.pt'))
    positions = test_case['positions'].to(device)
    positions.requires_grad = True
    cell = test_case['cell']
    if cell is not None:
        cell = cell.to(device)

    expected = test_case['output'].to(device)
    module = ANISymmetryFunctions(**test_case['parameters'])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = torch.concat(module(positions, cell), dim=1)
        total = torch.sum(actual)
        total.backward()
    torch.cuda.current_stream().wait_stream(stream)

    assert torch.allclose(actual, expected, rtol=5e-5)
    assert torch.allclose(positions.grad, test_case['grad'].to(device), rtol=GRADIENT_TOL, atol=GRADIENT_TOL)
