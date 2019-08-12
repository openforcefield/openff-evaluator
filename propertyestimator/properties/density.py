"""
A collection of density physical property definitions.
"""

from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties import PhysicalProperty, PropertyPhase
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.protocols import analysis
from propertyestimator.protocols.utils import generate_base_simulation_protocols, generate_base_reweighting_protocols, \
    generate_gradient_protocol_group
from propertyestimator.storage import StoredSimulationData
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@register_estimable_property()
@register_thermoml_property(thermoml_string='Mass density, kg/m3', supported_phases=PropertyPhase.Liquid)
class Density(PhysicalProperty):
    """A class representation of a density property"""

    @property
    def multi_component_property(self):
        return False

    @property
    def required_data_class(self):
        return StoredSimulationData

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return Density.get_default_simulation_workflow_schema(options)
        elif calculation_layer == 'ReweightingLayer':
            return Density.get_default_reweighting_workflow_schema(options)

        return None

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        from direct simulations.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Define the protocol which will extract the average density from
        # the results of a simulation.
        extract_density = analysis.ExtractAverageStatistic('extract_density')
        extract_density.statistics_type = ObservableType.Density

        # Define the protocols which will run the simulation itself.
        protocols, value_source, output_to_store = generate_base_simulation_protocols(extract_density,
                                                                                      options)

        # Set up the gradient calculations
        extract_raw_density = analysis.ExtractStatistic('extract_raw_density')
        extract_raw_density.statistics_type = ObservableType.Density
        extract_raw_density.statistics_path = ProtocolPath('statistics_file_path',
                                                           protocols.converge_uncertainty.id,
                                                           protocols.production_simulation.id)

        coordinate_source = ProtocolPath('output_coordinate_file', protocols.equilibration_simulation.id)
        trajectory_source = ProtocolPath('trajectory_file_path',
                                         protocols.converge_uncertainty.id,
                                         protocols.production_simulation.id)
        observables_source = ProtocolPath('values', extract_raw_density.id)

        gradient_group, gradient_replicator, gradient_source = \
            generate_gradient_protocol_group([ProtocolPath('force_field_path', 'global')],
                                             ProtocolPath('force_field_path', 'global'),
                                             coordinate_source,
                                             trajectory_source,
                                             observable_values=observables_source)

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=Density.__name__)
        schema.id = '{}{}'.format(Density.__name__, 'Schema')

        schema.protocols = {
            protocols.build_coordinates.id: protocols.build_coordinates.schema,
            protocols.assign_parameters.id: protocols.assign_parameters.schema,
            protocols.energy_minimisation.id: protocols.energy_minimisation.schema,
            protocols.equilibration_simulation.id: protocols.equilibration_simulation.schema,
            protocols.converge_uncertainty.id: protocols.converge_uncertainty.schema,
            protocols.extract_uncorrelated_trajectory.id: protocols.extract_uncorrelated_trajectory.schema,
            protocols.extract_uncorrelated_statistics.id: protocols.extract_uncorrelated_statistics.schema,
            extract_raw_density.id: extract_raw_density.schema,
            gradient_group.id: gradient_group.schema
        }

        schema.replicators = [gradient_replicator]

        schema.outputs_to_store = {'full_system': output_to_store}

        schema.gradients_sources = [gradient_source]
        schema.final_value_source = value_source

        return schema

    @staticmethod
    def get_default_reweighting_workflow_schema(options):
        """Returns the default workflow to use when estimating this property
        by reweighting existing data.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # The protocol which will be used to calculate the densities from
        # the existing data.
        density_calculation = analysis.ExtractAverageStatistic('calc_density_$(data_repl)')
        base_reweighting_protocols, data_replicator = generate_base_reweighting_protocols(density_calculation,
                                                                                          options)

        density_calculation.statistics_type = ObservableType.Density
        density_calculation.statistics_path = ProtocolPath('statistics_file_path',
                                                           base_reweighting_protocols.unpack_stored_data.id)

        # Set up the gradient calculations
        coordinate_path = ProtocolPath('output_coordinate_path', base_reweighting_protocols.concatenate_trajectories.id)
        trajectory_path = ProtocolPath('output_trajectory_path', base_reweighting_protocols.concatenate_trajectories.id)

        gradient_group, gradient_replicator, gradient_source = \
            generate_gradient_protocol_group([ProtocolPath('force_field_path',
                                                           base_reweighting_protocols.unpack_stored_data.id)],
                                             ProtocolPath('force_field_path', 'global'),
                                             coordinate_path,
                                             trajectory_path,
                                             'grad',
                                             ProtocolPath('uncorrelated_values', density_calculation.id),
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   base_reweighting_protocols.
                                                                                   mbar_protocol.id))

        schema = WorkflowSchema(property_type=Density.__name__)
        schema.id = '{}{}'.format(Density.__name__, 'Schema')

        schema.protocols = {protocol.id: protocol.schema for protocol in base_reweighting_protocols}
        schema.protocols[gradient_group.id] = gradient_group.schema

        schema.replicators = [data_replicator, gradient_replicator]

        schema.gradients_sources = [gradient_source]
        schema.final_value_source = ProtocolPath('value', base_reweighting_protocols.mbar_protocol.id)

        return schema
