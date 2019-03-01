"""
Units tests for propertyestimator.workflow
"""
import tempfile
from os import path

import pytest
from simtk import unit

from propertyestimator.backends import ComputeResources
from propertyestimator.properties.dielectric import ExtractAverageDielectric
from propertyestimator.substances import Mixture
from propertyestimator.tests.test_workflow.utils import DummyEstimatedQuantityProtocol, DummyProtocolWithDictInput
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow.plugins import available_protocols
from propertyestimator.workflow.protocols import AddQuantities, BuildCoordinatesPackmol, BuildSmirnoffSystem, \
    RunEnergyMinimisation, RunOpenMMSimulation, ExtractAverageStatistic, ExtractUncorrelatedTrajectoryData, \
    ExtractUncorrelatedStatisticsData, SubtractQuantities
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

    add_values_protocol = AddQuantities('add_values')

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

    add_values_protocol_2 = AddQuantities('add_values')

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

    # substance = Mixture()
    #
    # substance.add_component(smiles='C', mole_fraction=0.5)
    # substance.add_component(smiles='O', mole_fraction=0.5)
    #
    # build_coordinates = BuildCoordinatesPackmol('build_coordinates')
    # build_coordinates.substance = substance
    #
    # components = build_coordinates.get_value(ProtocolPath('substance.components', build_coordinates.id))
    # assert isinstance(components, list) and len(components) == 2
    #
    # fist_component = build_coordinates.get_value(ProtocolPath('substance.components[0]', build_coordinates.id))
    # assert isinstance(fist_component, Mixture.MixtureComponent) and fist_component.smiles == 'C'
    #
    # second_component = build_coordinates.get_value(ProtocolPath('substance.components[1]', build_coordinates.id))
    # assert isinstance(fist_component, Mixture.MixtureComponent) and fist_component.smiles == 'O'


def test_base_simulation_protocols():
    """Tests that the commonly chain build coordinates, assigned topology,
    energy minimise and perform simulation are able to work together without
    raising an exception."""

    mixed_system = Mixture()
    mixed_system.add_component(smiles='O', mole_fraction=1.0)

    thermodynamic_state = ThermodynamicState(298*unit.kelvin, 1*unit.atmosphere)

    with tempfile.TemporaryDirectory() as temporary_directory:

        build_coordinates = BuildCoordinatesPackmol('')

        # Set the maximum number of molecules in the system.
        build_coordinates.max_molecules = 10
        # and the target density (the default 1.0 g/ml is normally fine)
        build_coordinates.mass_density = 0.05 * unit.grams / unit.milliliters
        # and finally the system which coordinates should be generated for.
        build_coordinates.substance = mixed_system

        # Build the coordinates, creating a file called output.pdb
        result = build_coordinates.execute(temporary_directory, None)
        assert not isinstance(result, PropertyEstimatorException)

        # Assign some smirnoff force field parameters to the
        # coordinates
        print('Assigning some parameters.')
        assign_force_field_parameters = BuildSmirnoffSystem('')

        assign_force_field_parameters.force_field_path = get_data_filename('forcefield/smirnoff99Frosst.offxml')
        assign_force_field_parameters.coordinate_file_path = path.join(temporary_directory, 'output.pdb')
        assign_force_field_parameters.substance = mixed_system

        result = assign_force_field_parameters.execute(temporary_directory, None)
        assert not isinstance(result, PropertyEstimatorException)

        # Do a simple energy minimisation
        print('Performing energy minimisation.')
        energy_minimisation = RunEnergyMinimisation('')

        energy_minimisation.input_coordinate_file = path.join(temporary_directory, 'output.pdb')
        energy_minimisation.system = assign_force_field_parameters.system

        result = energy_minimisation.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        npt_equilibration = RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 20  # Debug settings.
        npt_equilibration.output_frequency = 2  # Debug settings.

        npt_equilibration.thermodynamic_state = thermodynamic_state

        npt_equilibration.input_coordinate_file = path.join(temporary_directory, 'minimised.pdb')
        npt_equilibration.system = assign_force_field_parameters.system

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
        extract_dielectric.system = assign_force_field_parameters.system

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

        add_quantities = AddQuantities('add')
        add_quantities.values = [quantity_a, quantity_b]

        result = add_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert add_quantities.result.value == 3 * unit.kelvin

        sub_quantities = SubtractQuantities('sub')
        sub_quantities.value_b = quantity_b
        sub_quantities.value_a = quantity_a

        result = sub_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert sub_quantities.result.value == 1 * unit.kelvin
