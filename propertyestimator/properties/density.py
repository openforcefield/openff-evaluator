"""
A collection of density physical property definitions.
"""

from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.properties import PhysicalProperty
from propertyestimator.storage import StoredSimulationData
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.utils.serialization import PolymorphicDataType
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow import protocols, groups
from propertyestimator.workflow.schemas import WorkflowOutputToStore
from propertyestimator.workflow.utils import ProtocolPath


@register_estimable_property()
@register_thermoml_property(thermoml_string='Mass density, kg/m3')
class Density(PhysicalProperty):
    """A class representation of a density property"""

    @staticmethod
    def get_default_calculation_schema():

        schema = WorkflowSchema(property_type=Density.__name__)
        schema.id = '{}{}'.format(Density.__name__, 'Schema')

        # Initial coordinate and topology setup.
        build_coordinates = protocols.BuildCoordinatesPackmol('build_coordinates')

        build_coordinates.substance = ProtocolPath('substance', 'global')

        schema.protocols[build_coordinates.id] = build_coordinates.schema

        assign_topology = protocols.BuildSmirnoffSystem('build_topology')

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

        npt_equilibration.steps = 2  # Debug settings.
        npt_equilibration.output_frequency = 2  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system = ProtocolPath('system', assign_topology.id)

        schema.protocols[npt_equilibration.id] = npt_equilibration.schema

        # Production
        npt_production = protocols.RunOpenMMSimulation('npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 200  # Debug settings.
        npt_production.output_frequency = 20  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system = ProtocolPath('system', assign_topology.id)

        # Analysis
        extract_density = protocols.ExtractAverageStatistic('extract_density')

        extract_density.statistics_type = ObservableType.Density
        extract_density.statistics_path = ProtocolPath('statistics_file_path', npt_production.id)

        # Set up a conditional group to ensure convergence of uncertainty
        converge_uncertainty = groups.ConditionalGroup('converge_uncertainty')
        converge_uncertainty.add_protocols(npt_production, extract_density)

        condition = groups.ConditionalGroup.Condition()

        condition.left_hand_value = ProtocolPath('value',
                                                 converge_uncertainty.id,
                                                 extract_density.id)

        condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

        condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

        converge_uncertainty.add_condition(condition)

        converge_uncertainty.max_iterations = 1

        schema.protocols[converge_uncertainty.id] = converge_uncertainty.schema

        # Finally, extract uncorrelated data
        extract_uncorrelated_trajectory = protocols.ExtractUncorrelatedTrajectoryData('extract_traj')

        extract_uncorrelated_trajectory.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                                converge_uncertainty.id,
                                                                                extract_density.id)

        extract_uncorrelated_trajectory.equilibration_index = ProtocolPath('equilibration_index',
                                                                           converge_uncertainty.id,
                                                                           extract_density.id)

        extract_uncorrelated_trajectory.input_coordinate_file = ProtocolPath('output_coordinate_file',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        extract_uncorrelated_trajectory.input_trajectory_path = ProtocolPath('trajectory_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        schema.protocols[extract_uncorrelated_trajectory.id] = extract_uncorrelated_trajectory.schema

        extract_uncorrelated_statistics = protocols.ExtractUncorrelatedStatisticsData('extract_stats')

        extract_uncorrelated_statistics.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                                converge_uncertainty.id,
                                                                                extract_density.id)

        extract_uncorrelated_statistics.equilibration_index = ProtocolPath('equilibration_index',
                                                                           converge_uncertainty.id,
                                                                           extract_density.id)

        extract_uncorrelated_statistics.input_statistics_path = ProtocolPath('statistics_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        schema.protocols[extract_uncorrelated_statistics.id] = extract_uncorrelated_statistics.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath('value', converge_uncertainty.id, extract_density.id)

        output_to_store = WorkflowOutputToStore()

        output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                            extract_uncorrelated_trajectory.id)
        output_to_store.coordinate_file_path = ProtocolPath('output_coordinate_file',
                                                            converge_uncertainty.id, npt_production.id)

        output_to_store.statistics_file_path = ProtocolPath('output_statistics_path',
                                                            extract_uncorrelated_statistics.id)

        output_to_store.statistical_inefficiency = ProtocolPath('statistical_inefficiency', converge_uncertainty.id,
                                                                                            extract_density.id)

        schema.outputs_to_store = {'full_system': PolymorphicDataType(output_to_store)}

        return schema

    @staticmethod
    def reweight(reference_data, target_property, target_force_field_id,
                 existing_data, existing_force_fields):

        """A placeholder method that would be used to attempt
        to reweight previous calculations to yield the desired
        property.

        Warnings
        --------
        This method has not yet been implemented.

        Parameters
        ----------
        physical_property: :obj:`propertyestimator.properties.PhysicalProperty`
            The physical property to attempt to estimate by reweighting.
        parameter_set_id: :obj:`str`
            The id of the force field parameters which the property should be
            estimated with.
        existing_data: :obj:`dict` of :obj:`str` and :obj:`propertyestimator.storage.StoredSimulationData`
            Data which has been stored from previous calculations on systems
            of the same composition as the desired property.
        existing_force_fields: :obj:`dict` of :obj:`str` and :obj:`dict` of :obj:`int` and :obj:`str`
            A dictionary of all of the force field parameters referenced by the
            `existing_data`, which have been serialized with `serialize_force_field`
        """

        target_force_field = deserialize_force_field(existing_force_fields[parameter_set_id])

        particle_counts = np.array([data.trajectory_data.n_chains for data in existing_data])
        maximum_molecule_count = particle_counts.max()

        # Only retain data which has the same number of molecules. For now
        # we choose the data which was calculated using the most molecules,
        # however perhaps we should instead choose data with the mode number
        # of molecules?
        useable_data = [data for data in existing_data if
                        data.trajectory_data.n_chains == maximum_molecule_count]

        for data in useable_data:

            # Calculate the number of uncorrelated samples per data object.
            frame_counts = np.array([data.trajectory_data.n_frames]*2)

            reduced_energies = np.zeros(2, frame_counts.max())
            observables = np.zeros(2, frame_counts.max())

            # The reference state energies.
            reduced_energies[0] = ReweightingLayer._get_reduced_reference_energies(physical_property, data)

            # The target state energies.
            reduced_energies[1] = ReweightingLayer._get_reduced_target_energies(physical_property, parameter_set_id,
                                                                                target_force_field, data)

            property_class = registered_properties[physical_property.type]

            # Calculate the observables.
            # observables[1] = property_class.calculate_observable(data)

            mbar = pymbar.MBAR(reduced_energies, frame_counts, verbose=False, relative_tolerance=1e-12)
            results = mbar.computeExpectations(observables, state_dependent=True)

            all_values = results['mu']
            all_uncertainties = results['sigma']

            if all_uncertainties[1] < physical_property.uncertainty:

                physical_property.value = all_values[1]
                physical_property.uncertainty = all_uncertainties[1]

                physical_property.source = CalculationSource()

                physical_property.source.fidelity = ReweightingLayer.__name__
                physical_property.source.provenance = {
                    'data_sources': []  # TODO: Add tags to data sources
                }

                return physical_property

        return None