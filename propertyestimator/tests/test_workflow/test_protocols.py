"""
Units tests for propertyestimator.workflow
"""
import tempfile
from os import path
from tempfile import NamedTemporaryFile

import pytest
from openforcefield.topology import Molecule, Topology
from simtk.openmm.app import PDBFile

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.properties.dielectric import ExtractAverageDielectric
from propertyestimator.protocols.analysis import ExtractAverageStatistic, ExtractUncorrelatedTrajectoryData, \
    ExtractUncorrelatedStatisticsData
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol, SolvateExistingStructure
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.forcefield import BuildTLeapSystem
from propertyestimator.protocols.miscellaneous import AddValues, FilterSubstanceByRole, SubtractValues
from propertyestimator.protocols.simulation import RunEnergyMinimisation, RunOpenMMSimulation
from propertyestimator.substances import Substance
from propertyestimator.tests.test_workflow.utils import DummyEstimatedQuantityProtocol, DummyProtocolWithDictInput
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.utils.utils import get_data_filename
from propertyestimator.workflow.plugins import available_protocols
from propertyestimator.workflow.utils import ProtocolPath


@pytest.mark.parametrize("available_protocol", available_protocols)
def test_default_protocol_schemas(available_protocol):
    """A simple test to ensure that each available protocol
    can both create, and be created from a schema."""
    protocol = available_protocols[available_protocol]('dummy_id')
    protocol_schema = protocol.schema

    recreated_protocol = available_protocols[available_protocol]('dummy_id')
    recreated_protocol.schema = protocol_schema

    assert protocol.schema.json() == recreated_protocol.schema.json()


def test_nested_protocol_paths():

    value_protocol_a = DummyEstimatedQuantityProtocol('protocol_a')
    value_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'constant')

    assert value_protocol_a.get_value(ProtocolPath('input_value.value')) == value_protocol_a.input_value.value

    value_protocol_a.set_value(ProtocolPath('input_value._value'), 0.5*unit.kelvin)
    assert value_protocol_a.input_value.value == 0.5*unit.kelvin

    value_protocol_b = DummyEstimatedQuantityProtocol('protocol_b')
    value_protocol_b.input_value = EstimatedQuantity(2 * unit.kelvin, 0.05 * unit.kelvin, 'constant')

    value_protocol_c = DummyEstimatedQuantityProtocol('protocol_c')
    value_protocol_c.input_value = EstimatedQuantity(4 * unit.kelvin, 0.01 * unit.kelvin, 'constant')

    add_values_protocol = AddValues('add_values')

    add_values_protocol.values = [
        ProtocolPath('output_value', value_protocol_a.id),
        ProtocolPath('output_value', value_protocol_b.id),
        ProtocolPath('output_value', value_protocol_b.id),
        5
    ]

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('valus[string]'))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('values[string]'))

    input_values = add_values_protocol.get_value_references(ProtocolPath('values'))
    assert isinstance(input_values, dict) and len(input_values) == 3

    for index, value_reference in enumerate(input_values):

        input_value = add_values_protocol.get_value(value_reference)
        assert input_value.full_path == add_values_protocol.values[index].full_path

        add_values_protocol.set_value(value_reference, index)

    assert set(add_values_protocol.values) == {0, 1, 2, 5}

    dummy_dict_protocol = DummyProtocolWithDictInput('dict_protocol')

    dummy_dict_protocol.input_value = {
        'value_a': ProtocolPath('output_value', value_protocol_a.id),
        'value_b': ProtocolPath('output_value', value_protocol_b.id),
    }

    input_values = dummy_dict_protocol.get_value_references(ProtocolPath('input_value'))
    assert isinstance(input_values, dict) and len(input_values) == 2

    for index, value_reference in enumerate(input_values):

        input_value = dummy_dict_protocol.get_value(value_reference)

        dummy_dict_keys = list(dummy_dict_protocol.input_value.keys())
        assert input_value.full_path == dummy_dict_protocol.input_value[dummy_dict_keys[index]].full_path

        dummy_dict_protocol.set_value(value_reference, index)

    add_values_protocol_2 = AddValues('add_values')

    add_values_protocol_2.values = [
        [ProtocolPath('output_value', value_protocol_a.id)],
        [
            ProtocolPath('output_value', value_protocol_b.id),
            ProtocolPath('output_value', value_protocol_b.id)
        ]
    ]

    with pytest.raises(ValueError):
        add_values_protocol_2.get_value(ProtocolPath('valus[string]'))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('values[string]'))

    pass


def test_base_simulation_protocols():
    """Tests that the commonly chain build coordinates, assigned topology,
    energy minimise and perform simulation are able to work together without
    raising an exception."""

    water_substance = Substance()
    water_substance.add_component(Substance.Component(smiles='O'),
                                  Substance.MoleFraction())

    thermodynamic_state = ThermodynamicState(298*unit.kelvin, 1*unit.atmosphere)

    with tempfile.TemporaryDirectory() as temporary_directory:

        force_field_source = build_tip3p_smirnoff_force_field()
        force_field_path = path.join(temporary_directory, 'ff.offxml')

        with open(force_field_path, 'w') as file:
            file.write(force_field_source.json())

        build_coordinates = BuildCoordinatesPackmol('')

        # Set the maximum number of molecules in the system.
        build_coordinates.max_molecules = 10
        # and the target density (the default 1.0 g/ml is normally fine)
        build_coordinates.mass_density = 0.05 * unit.grams / unit.milliliters
        # and finally the system which coordinates should be generated for.
        build_coordinates.substance = water_substance

        # Build the coordinates, creating a file called output.pdb
        result = build_coordinates.execute(temporary_directory, None)
        assert not isinstance(result, PropertyEstimatorException)

        # Assign some smirnoff force field parameters to the
        # coordinates
        print('Assigning some parameters.')
        assign_force_field_parameters = BuildSmirnoffSystem('')

        assign_force_field_parameters.force_field_path = force_field_path
        assign_force_field_parameters.coordinate_file_path = path.join(temporary_directory, 'output.pdb')
        assign_force_field_parameters.substance = water_substance

        result = assign_force_field_parameters.execute(temporary_directory, None)
        assert not isinstance(result, PropertyEstimatorException)

        # Do a simple energy minimisation
        print('Performing energy minimisation.')
        energy_minimisation = RunEnergyMinimisation('')

        energy_minimisation.input_coordinate_file = path.join(temporary_directory, 'output.pdb')
        energy_minimisation.system_path = assign_force_field_parameters.system_path

        result = energy_minimisation.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        npt_equilibration = RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 20  # Debug settings.
        npt_equilibration.output_frequency = 2  # Debug settings.

        npt_equilibration.thermodynamic_state = thermodynamic_state

        npt_equilibration.input_coordinate_file = path.join(temporary_directory, 'minimised.pdb')
        npt_equilibration.system_path = assign_force_field_parameters.system_path

        result = npt_equilibration.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        extract_density = ExtractAverageStatistic('extract_density')

        extract_density.statistics_type = ObservableType.Density
        extract_density.statistics_path = path.join(temporary_directory, 'statistics.csv')

        result = extract_density.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        extract_dielectric = ExtractAverageDielectric('extract_dielectric')

        extract_dielectric.thermodynamic_state = thermodynamic_state

        extract_dielectric.input_coordinate_file = path.join(temporary_directory, 'input.pdb')
        extract_dielectric.trajectory_path = path.join(temporary_directory, 'trajectory.dcd')
        extract_dielectric.system_path = assign_force_field_parameters.system_path

        result = extract_dielectric.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        extract_uncorrelated_trajectory = ExtractUncorrelatedTrajectoryData('extract_traj')

        extract_uncorrelated_trajectory.statistical_inefficiency = extract_density.statistical_inefficiency
        extract_uncorrelated_trajectory.equilibration_index = extract_density.equilibration_index
        extract_uncorrelated_trajectory.input_coordinate_file = path.join(temporary_directory, 'input.pdb')
        extract_uncorrelated_trajectory.input_trajectory_path = path.join(temporary_directory, 'trajectory.dcd')

        result = extract_uncorrelated_trajectory.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        extract_uncorrelated_statistics = ExtractUncorrelatedStatisticsData('extract_stats')

        extract_uncorrelated_statistics.statistical_inefficiency = extract_density.statistical_inefficiency
        extract_uncorrelated_statistics.equilibration_index = extract_density.equilibration_index
        extract_uncorrelated_statistics.input_statistics_path = path.join(temporary_directory, 'statistics.csv')

        result = extract_uncorrelated_statistics.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)


def test_addition_subtract_protocols():

    with tempfile.TemporaryDirectory() as temporary_directory:

        quantity_a = EstimatedQuantity(1*unit.kelvin, 0.1*unit.kelvin, 'dummy_source_1')
        quantity_b = EstimatedQuantity(2*unit.kelvin, 0.2*unit.kelvin, 'dummy_source_2')

        add_quantities = AddValues('add')
        add_quantities.values = [quantity_a, quantity_b]

        result = add_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert add_quantities.result.value == 3 * unit.kelvin

        sub_quantities = SubtractValues('sub')
        sub_quantities.value_b = quantity_b
        sub_quantities.value_a = quantity_a

        result = sub_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert sub_quantities.result.value == 1 * unit.kelvin


@pytest.mark.parametrize("filter_role", [Substance.ComponentRole.Solute,
                                         Substance.ComponentRole.Solvent,
                                         Substance.ComponentRole.Ligand,
                                         Substance.ComponentRole.Receptor])
def test_substance_filtering_protocol(filter_role):
    """Tests that the protocol to filter substances by
    role correctly works."""

    def create_substance():

        test_substance = Substance()

        test_substance.add_component(Substance.Component('C', role=Substance.ComponentRole.Solute),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('CC', role=Substance.ComponentRole.Ligand),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('CCC', role=Substance.ComponentRole.Receptor),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('O', role=Substance.ComponentRole.Solvent),
                                     Substance.MoleFraction(1.0))

        return test_substance

    filter_protocol = FilterSubstanceByRole('filter_protocol')
    filter_protocol.input_substance = create_substance()

    filter_protocol.component_role = filter_role
    filter_protocol.execute('', ComputeResources())

    assert len(filter_protocol.filtered_substance.components) == 1
    assert filter_protocol.filtered_substance.components[0].role == filter_role


def _build_input_output_substances():
    """Builds sets if input and expected substances for the
    `test_build_coordinate_composition` test.

    Returns
    -------
    list of tuple of Substance and Substance
        A list of input and expected substances.
    """

    # Start with some easy cases
    substances = [
        (Substance.from_components('O'), Substance.from_components('O')),
        (Substance.from_components('O', 'C'), Substance.from_components('O', 'C')),
        (Substance.from_components('O', 'C', 'CO'), Substance.from_components('O', 'C', 'CO'))
    ]

    # Handle some cases where rounding will need to occur.
    input_substance = Substance()
    input_substance.add_component(Substance.Component('O'), Substance.MoleFraction(0.41))
    input_substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.59))

    expected_substance = Substance()
    expected_substance.add_component(Substance.Component('O'), Substance.MoleFraction(0.4))
    expected_substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.6))

    substances.append((input_substance, expected_substance))

    input_substance = Substance()
    input_substance.add_component(Substance.Component('O'), Substance.MoleFraction(0.59))
    input_substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.41))

    expected_substance = Substance()
    expected_substance.add_component(Substance.Component('O'), Substance.MoleFraction(0.6))
    expected_substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.4))

    substances.append((input_substance, expected_substance))

    return substances


@pytest.mark.parametrize("input_substance, expected", _build_input_output_substances())
def test_build_coordinate_composition(input_substance, expected):
    """Tests that the build coordinate protocols correctly report
    the composition of the built system."""

    build_coordinates = BuildCoordinatesPackmol('build_coordinates')
    build_coordinates.max_molecules = 10
    build_coordinates.substance = input_substance

    with tempfile.TemporaryDirectory() as directory:
        assert not isinstance(build_coordinates.execute(directory, None), PropertyEstimatorException)

    assert build_coordinates.output_substance == expected


def test_solvation_protocol():
    """Tests solvating a single methanol molecule in water."""

    methanol_substance = Substance()
    methanol_substance.add_component(Substance.Component('CO'), Substance.ExactAmount(1))

    water_substance = Substance()
    water_substance.add_component(Substance.Component('O'), Substance.MoleFraction(1.0))

    with tempfile.TemporaryDirectory() as temporary_directory:

        build_methanol_coordinates = BuildCoordinatesPackmol('build_methanol')

        build_methanol_coordinates.max_molecules = 1
        build_methanol_coordinates.substance = methanol_substance

        build_methanol_coordinates.execute(temporary_directory, ComputeResources())

        solvate_coordinates = SolvateExistingStructure('solvate_methanol')

        solvate_coordinates.max_molecules = 9
        solvate_coordinates.substance = water_substance
        solvate_coordinates.solute_coordinate_file = build_methanol_coordinates.coordinate_file_path

        solvate_coordinates.execute(temporary_directory, ComputeResources())

        solvated_pdb = PDBFile(solvate_coordinates.coordinate_file_path)

        assert solvated_pdb.topology.getNumResidues() == 10


@pytest.mark.parametrize('toolkit_name', ['OpenEye', 'RDKit'])
def test_topology_mol_to_mol2(toolkit_name):
    """Tests taking an openforcefield topology molecule, generating a conformer,
    calculating partial charges, and writing it to mol2."""

    # Testing to find the correct connectivity information should indicate 
    # that the mol2 conversion was successful
    expected_contents = ['''
@<TRIPOS>BOND
     1    1    3 1
     2    2    3 1
     3    3    9 1
     4    3    8 1
     5    1    4 1
     6    2    5 1
     7    2    6 1
     8    2    7 1''', '''@<TRIPOS>BOND
     1    4    6 ar
     2    2    4 ar
     3    1    2 ar
     4    1    3 ar
     5    3    5 ar
     6    5    6 ar
     7    6    7 1
     8    4   11 1
     9    2    9 1
    10    1    8 1
    11    3   10 1
    12    5   12 1
    13    7   13 1
    14    7   14 1
    15    7   15 1''']

    # Constructing the molecule using this SMILES will ensure that the reference molecule
    # (generated here) and topology molecule (generated below from PDB) have a different atom order
    ethanol_smiles = 'C(O)C'
    toluene_smiles = 'c1ccccc1C'
    
    ethanol = Molecule.from_smiles(ethanol_smiles)
    toluene = Molecule.from_smiles(toluene_smiles)

    pdb_file = PDBFile(get_data_filename('test/molecules/ethanol_toluene.pdb'))
    topology = Topology.from_openmm(pdb_file.topology, unique_molecules=[ethanol, toluene])

    for topology_molecule_index, topology_molecule in enumerate(topology.topology_molecules):

        with NamedTemporaryFile(suffix='.mol2') as output_file:

            if toolkit_name is None:

                # Because the molecules come from SMILES, they don't come with any
                # coords or charges
                with pytest.raises(TypeError):

                    BuildTLeapSystem._topology_molecule_to_mol2(topology_molecule,
                                                                output_file.name,
                                                                toolkit=toolkit_name)

                continue

            else:

                BuildTLeapSystem._topology_molecule_to_mol2(topology_molecule,
                                                            output_file.name,
                                                            toolkit=toolkit_name)

                mol2_contents = open(output_file.name).read()

                # Ensure we find the correct connectivity info
                assert expected_contents[topology_molecule_index] in mol2_contents

                # Ensure that the first atom has nonzero coords and charge
                first_atom_line = mol2_contents.split('\n')[7].split()
                assert float(first_atom_line[2]) != 0.
                assert float(first_atom_line[3]) != 0.
                assert float(first_atom_line[4]) != 0.
                assert float(first_atom_line[8]) != 0.
