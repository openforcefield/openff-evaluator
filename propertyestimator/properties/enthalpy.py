"""
A collection of enthalpy physical property definitions.
"""

from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.properties import PhysicalProperty
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.utils.statistics import AvailableQuantities
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow import protocols, groups
from propertyestimator.workflow.utils import ProtocolPath, PlaceholderInput
from propertyestimator.workflow.plugins import available_protocols


@register_estimable_property()
class Enthalpy(PhysicalProperty):
    """A class representation of a enthalpy property"""

    @staticmethod
    def get_default_calculation_schema():

        schema = WorkflowSchema(property_type=Enthalpy.__name__)
        schema.id = '{}{}'.format(Enthalpy.__name__, 'Schema')

        # Initial coordinate and topology setup.
        build_coordinates = protocols.BuildCoordinatesPackmol('build_coordinates')

        build_coordinates.substance = ProtocolPath('substance', 'global')

        schema.protocols[build_coordinates.id] = build_coordinates.schema

        assign_topology = protocols.BuildSmirnoffTopology('build_topology')

        assign_topology.force_field_path = ProtocolPath('force_field_path', 'global')

        assign_topology.coordinate_file_path = ProtocolPath('coordinate_file_path', build_coordinates.id)
        assign_topology.substance = ProtocolPath('substance', 'global')

        schema.protocols[assign_topology.id] = assign_topology.schema

        # Equilibration
        energy_minimisation = protocols.RunEnergyMinimisation('energy_minimisation')

        energy_minimisation.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
        energy_minimisation.system = ProtocolPath('system', assign_topology.id)

        schema.protocols[energy_minimisation.id] = energy_minimisation.schema

        npt_equilibration = protocols.RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 2000  # Debug settings.
        npt_equilibration.output_frequency = 200  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system = ProtocolPath('system', assign_topology.id)

        schema.protocols[npt_equilibration.id] = npt_equilibration.schema

        # Production
        npt_production = protocols.RunOpenMMSimulation('npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 20000  # Debug settings.
        npt_production.output_frequency = 200  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system = ProtocolPath('system', assign_topology.id)

        # Analysis
        # extract_enthalpy = ExtractAverageEnthalpy('extract_enthalpy')
        #
        # extract_enthalpy.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_production.id)
        # extract_enthalpy.trajectory_path = ProtocolPath('trajectory_file_path', npt_production.id)
        # extract_enthalpy.system = ProtocolPath('system', assign_topology.id)

        extract_enthalpy = protocols.ExtractAverageStatistic('extract_enthalpy')

        extract_enthalpy.statistics_type = AvailableQuantities.Enthalpy
        extract_enthalpy.statistics_path = ProtocolPath('statistics_file_path', npt_production.id)

        # Set up a conditional group to ensure convergence of uncertainty
        converge_uncertainty = groups.ConditionalGroup('converge_uncertainty')
        converge_uncertainty.add_protocols(npt_production, extract_enthalpy)

        condition = groups.ConditionalGroup.Condition()

        condition.left_hand_value = ProtocolPath('uncertainty',
                                                 converge_uncertainty.id,
                                                 extract_enthalpy.id)

        condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

        condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

        converge_uncertainty.add_condition(condition)

        converge_uncertainty.max_iterations = 10

        schema.protocols[converge_uncertainty.id] = converge_uncertainty.schema

        # Finally, extract uncorrelated data
        extract_uncorrelated_trajectory = protocols.ExtractUncorrelatedTrajectoryData('extract_traj')

        extract_uncorrelated_trajectory.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                                converge_uncertainty.id,
                                                                                extract_enthalpy.id)

        extract_uncorrelated_trajectory.equilibration_index = ProtocolPath('equilibration_index',
                                                                           converge_uncertainty.id,
                                                                           extract_enthalpy.id)

        extract_uncorrelated_trajectory.input_coordinate_file = ProtocolPath('output_coordinate_file',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        extract_uncorrelated_trajectory.input_trajectory_path = ProtocolPath('trajectory_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        schema.protocols[extract_uncorrelated_trajectory.id] = extract_uncorrelated_trajectory.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath('value', converge_uncertainty.id, extract_enthalpy.id)
        schema.final_uncertainty_source = ProtocolPath('uncertainty', converge_uncertainty.id, extract_enthalpy.id)

        schema.final_coordinate_source = ProtocolPath('output_coordinate_file', converge_uncertainty.id,
                                                                                npt_production.id)

        schema.final_trajectory_source = ProtocolPath('output_trajectory_path', extract_uncorrelated_trajectory.id)

        return schema


@register_estimable_property()
@register_thermoml_property(thermoml_string='Excess molar enthalpy (molar enthalpy of mixing), kJ/mol')
class EnthalpyOfMixing(PhysicalProperty):
    """A class representation of a enthalpy property"""

    @staticmethod
    def get_enthalpy_workflow_schemas(id_prefix=''):

        build_coordinates = protocols.BuildCoordinatesPackmol(id_prefix + 'build_coordinates')

        build_coordinates.substance = ProtocolPath('substance', 'global')

        assign_topology = protocols.BuildSmirnoffTopology(id_prefix + 'build_topology')

        assign_topology.force_field_path = ProtocolPath('force_field_path', 'global')

        assign_topology.coordinate_file_path = ProtocolPath('coordinate_file_path', build_coordinates.id)
        assign_topology.substance = ProtocolPath('substance', 'global')

        # Equilibration
        energy_minimisation = protocols.RunEnergyMinimisation(id_prefix + 'energy_minimisation')

        energy_minimisation.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
        energy_minimisation.system = ProtocolPath('system', assign_topology.id)

        npt_equilibration = protocols.RunOpenMMSimulation(id_prefix + 'npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 2000  # Debug settings.
        npt_equilibration.output_frequency = 200  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system = ProtocolPath('system', assign_topology.id)

        # Production
        npt_production = protocols.RunOpenMMSimulation(id_prefix + 'npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 20000  # Debug settings.
        npt_production.output_frequency = 200  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system = ProtocolPath('system', assign_topology.id)

        # Analysis
        extract_enthalpy = protocols.ExtractAverageStatistic(id_prefix + 'extract_enthalpy')

        extract_enthalpy.statistics_type = AvailableQuantities.Enthalpy
        extract_enthalpy.statistics_path = ProtocolPath('statistics_file_path', npt_production.id)

        return {
            build_coordinates.id: build_coordinates.schema,
            assign_topology.id: assign_topology.schema,
            energy_minimisation.id: energy_minimisation.schema,
            npt_equilibration.id: npt_equilibration.schema,
            npt_production.id: npt_production.schema,
            extract_enthalpy.id: extract_enthalpy.schema
        }

    @staticmethod
    def get_default_calculation_schema():

        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        component_generator_group = groups.GeneratorGroup('component_generator')

        component_enthalpy_schemas = EnthalpyOfMixing.get_enthalpy_workflow_schemas('component_$index_')
        component_protocols = []

        for protocol_id in component_enthalpy_schemas:

            protocol_schema = component_enthalpy_schemas[protocol_id]

            protocol = available_protocols[protocol_schema.type]('')
            protocol.schema = protocol_schema

            if protocol_id.find('build_coordinates') >= 0 or protocol_id.find('build_topology') >= 0:
                protocol.substance = PlaceholderInput()

            component_protocols.append(protocol)

        component_generator_group.add_protocols(*component_protocols)

        component_generator_group.template_targets = [
            ProtocolPath('substance', 'build_coordinates'),
            ProtocolPath('substance', 'build_topology')
        ]

        component_generator_group.template_values = [
            ProtocolPath('components', 'global')
        ]

        schema.protocols[component_generator_group.id] = component_generator_group.schema

        full_enthalpy_schemas = EnthalpyOfMixing.get_enthalpy_workflow_schemas('full_system_')

        for protocol_id in full_enthalpy_schemas:
            schema.protocols[protocol_id] = full_enthalpy_schemas[protocol_id]

        return schema
