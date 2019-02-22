#!/usr/bin/env python
from simtk import unit

from propertyestimator.backends import ComputeResources
from propertyestimator.substances import Mixture
from propertyestimator.utils import get_data_filename
from propertyestimator.workflow.protocols import BuildCoordinatesPackmol, BuildSmirnoffTopology, RunEnergyMinimisation


def build_solvated_system():
    """An example of how to build a solvated system using the built in
    utilities and protocol classes.
    """

    # Define the system that you wish to create coordinates for.
    mixed_system = Mixture()

    # Here we simply define a 1:1 mix of water and octanol.
    mixed_system.add_component(smiles='O', mole_fraction=0.5)
    mixed_system.add_component(smiles='CCCCCCCCO', mole_fraction=0.5)

    # Add any 'impurities' such as single solute molecules.
    # In this case we add a molecule of paracetamol.
    mixed_system.add_component(smiles='CC(=O)NC1=CC=C(C=C1)O', mole_fraction=0.0, impurity=True)

    # Create an object which under the hood calls the packmol utility
    # in a friendlier way:
    print('Building the coordinates (this may take a while...)')

    build_coordinates = BuildCoordinatesPackmol('')

    # Set the maximum number of molecules in the system.
    build_coordinates.max_molecules = 1500
    # and the target density (the default 1.0 g/ml is normally fine)
    build_coordinates.mass_density = 1.0 * unit.grams / unit.milliliters
    # and finally the system which coordinates should be generated for.
    build_coordinates.substance = mixed_system

    # Build the coordinates, creating a file called output.pdb
    build_coordinates.execute('', None)

    # Assign some smirnoff force field parameters to the
    # coordinates
    print('Assigning some parameters.')
    assign_force_field_parameters = BuildSmirnoffTopology('')

    assign_force_field_parameters.force_field_path = get_data_filename('forcefield/smirnoff99Frosst.offxml')
    assign_force_field_parameters.coordinate_file_path = 'output.pdb'
    assign_force_field_parameters.substance = mixed_system

    assign_force_field_parameters.execute('', None)

    # Do a simple energy minimisation
    print('Performing energy minimisation.')
    energy_minimisation = RunEnergyMinimisation('')

    energy_minimisation.input_coordinate_file = 'output.pdb'
    energy_minimisation.system = assign_force_field_parameters.system

    energy_minimisation.execute('', ComputeResources())


if __name__ == "__main__":
    build_solvated_system()
