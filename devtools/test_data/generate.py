#!/usr/bin/env python

"""
Uses TorchANI to generate test cases in `src/pytorch/test_data/*.pt` for the
ANI symmetry functions.  Requires TorchANI and mdtraj to be installed.
"""

import mdtraj
import torch
import torchani

def main():
    for name in ("1hvj", "1hvk", "2iuz", "3hkw", "3hky", "3lka", "3o99"):
        generate_molecule_test_case(f"{name}_ligand.mol2", f"{name}.pt")
    generate_molecule_test_case("water.pdb", "water.pt", True)

def generate_molecule_test_case(in_path, out_path, pbc=False):
    molecule = mdtraj.load(in_path)
    atomic_numbers = torch.tensor([atom.element.atomic_number for atom in molecule.top.atoms])
    atomic_positions = torch.tensor(molecule.xyz[0] * 10, requires_grad=True)
    cell = torch.tensor(molecule.unitcell_vectors[0] * 10) if pbc else None

    nnp = torchani.models.ANI2x()

    parameters = dict(
        numSpecies=nnp.aev_computer.num_species,
        Rcr=nnp.aev_computer.radial.cutoff,
        Rca=nnp.aev_computer.angular.cutoff,
        EtaR=nnp.aev_computer.radial.eta.tolist(),
        ShfR=nnp.aev_computer.radial.shifts.tolist(),
        EtaA=nnp.aev_computer.angular.eta.tolist(),
        Zeta=nnp.aev_computer.angular.zeta.tolist(),
        ShfA=nnp.aev_computer.angular.shifts.tolist(),
        ShfZ=nnp.aev_computer.angular.sections.tolist(),
        atomSpecies=nnp.species_converter(atomic_numbers).tolist(),
    )
    output = nnp.aev_computer.forward(
        torch.tensor(parameters["atomSpecies"]).unsqueeze(0),
        atomic_positions.unsqueeze(0),
        cell,
        None if cell is None else torch.tensor([True, True, True])
    )[0]
    total = torch.sum(output)
    total.backward()
    testcase = dict(
        parameters=parameters,
        positions=atomic_positions.detach(),
        cell=cell,
        output=output,
        grad=atomic_positions.grad,
    )
    torch.save(testcase, out_path)

if __name__ == "__main__":
    main()
