"""
A collection of dielectric physical property definitions.
"""

import logging

import mdtraj
import numpy as np
from simtk import openmm, unit
from simtk.openmm import System

from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.properties import PhysicalProperty
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils import timeseries
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow import protocols, groups, plugins
from propertyestimator.workflow.decorators import protocol_input
from propertyestimator.workflow.schemas import WorkflowOutputToStore, ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


@plugins.register_calculation_protocol()
class ExtractAverageDielectric(protocols.AverageTrajectoryProperty):
    """Extracts the average dielectric constant from a simulation trajectory.
    """
    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._system = None
        self._thermodynamic_state = None

    @protocol_input(System)
    def system(self, value):
        """The system object which defines the forces present in the system."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self, value):
        """The thermodynamic state at which the trajectory was generated."""
        pass

    def _bootstrap_function(self, sample_data):
        """Calculates the static dielectric constant from an
        array of dipoles and volumes.

        Notes
        -----
        The static dielectric constant is taken from for Equation 7 of [1]

        References
        ----------
        [1] A. Glattli, X. Daura and W. F. van Gunsteren. Derivation of an improved simple point charge
            model for liquid water: SPC/A and SPC/L. J. Chem. Phys. 116(22):9811-9828, 2002

        Parameters
        ----------
        sample_data: np.ndarray, shape=(num_frames, 4), dtype=float
            The dataset to bootstap. The data stored by trajectory frame is
            four dimensional (Mx, My, Mz, V) where M is dipole moment and
            V is volume.

        Returns
        -------
        float
            The unitless static dielectric constant
        """

        temperature = self._thermodynamic_state.temperature

        dipoles = np.zeros([sample_data.shape[0], 3])
        volumes = np.zeros([sample_data.shape[0], 1])

        for index in range(sample_data.shape[0]):

            dipoles[index][0] = sample_data[index][0]
            dipoles[index][1] = sample_data[index][1]
            dipoles[index][2] = sample_data[index][2]

            volumes[index] = sample_data[index][3]

        dipole_mu = dipoles.mean(0)
        shifted_dipoles = dipoles - dipole_mu

        dipole_variance = (shifted_dipoles * shifted_dipoles).sum(-1).mean(0) * \
                          (unit.elementary_charge * unit.nanometers) ** 2

        volume = volumes.mean() * unit.nanometer**3

        e0 = 8.854187817E-12 * unit.farad / unit.meter  # Taken from QCElemental

        dielectric_constant = 1.0 + dipole_variance / (3 *
                                                       unit.BOLTZMANN_CONSTANT_kB *
                                                       temperature *
                                                       volume *
                                                       e0)

        return dielectric_constant

    def execute(self, directory, available_resources):

        logging.info('Extracting dielectrics: ' + self.id)

        base_exception = super(ExtractAverageDielectric, self).execute(directory, available_resources)

        if isinstance(base_exception, ExtractAverageDielectric):
            return base_exception

        charge_list = []

        for force_index in range(self._system.getNumForces()):

            force = self._system.getForce(force_index)

            if not isinstance(force, openmm.NonbondedForce):
                continue

            for atom_index in range(force.getNumParticles()):

                charge = force.getParticleParameters(atom_index)[0]
                charge /= unit.elementary_charge

                charge_list.append(charge)

        dipole_moments = mdtraj.geometry.dipole_moments(self.trajectory, charge_list)
        volumes = self.trajectory.unitcell_volumes

        dipole_moments, self._equilibration_index, self._statistical_inefficiency = \
            timeseries.decorrelate_time_series(dipole_moments)

        self._uncorrelated_values = unit.Quantity(dipole_moments, None)

        dipole_moments_and_volume = np.zeros([dipole_moments.shape[0], 4])

        for index in range(dipole_moments.shape[0]):

            dipole = dipole_moments[index]
            volume = volumes[index]

            dipole_moments_and_volume[index] = np.array([dipole[0], dipole[1], dipole[2], volume])

        value, uncertainty = self._perform_bootstrapping(dipole_moments_and_volume)

        self._value = EstimatedQuantity(unit.Quantity(value, None),
                                        unit.Quantity(uncertainty, None), self.id)

        logging.info('Extracted dielectrics: ' + self.id)

        return self._get_output_dictionary()


@register_estimable_property()
@register_thermoml_property(thermoml_string='Relative permittivity at zero frequency')
class DielectricConstant(PhysicalProperty):
    """A class representation of a dielectric property"""

    @property
    def multi_component_property(self):
        """Returns whether this property is dependant on properties of the
        full mixed substance, or whether it is also dependant on the properties
        of the individual components also.
        """
        return False

    @staticmethod
    def get_default_workflow_schema(calculation_layer):
        """Returns the default workflow schema to use for
        a specific calculation layer.

        Parameters
        ----------
        calculation_layer: str
            The calculation layer which will attempt to execute the workflow
            defined by this schema.

        Returns
        -------
        WorkflowSchema
            The default workflow schema.
        """
        if calculation_layer == 'SimulationLayer':
            return DielectricConstant.get_default_simulation_workflow_schema()
        elif calculation_layer == 'ReweightingLayer':
            return DielectricConstant.get_default_reweighting_workflow_schema()

        return None

    @staticmethod
    def get_default_simulation_workflow_schema():

        schema = WorkflowSchema(property_type=DielectricConstant.__name__)
        schema.id = '{}{}'.format(DielectricConstant.__name__, 'Schema')

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
        npt_equilibration.output_frequency = 1  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system = ProtocolPath('system', assign_topology.id)

        schema.protocols[npt_equilibration.id] = npt_equilibration.schema

        # Production
        npt_production = protocols.RunOpenMMSimulation('npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 20  # Debug settings.
        npt_production.output_frequency = 2  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system = ProtocolPath('system', assign_topology.id)

        # Analysis
        extract_dielectric = ExtractAverageDielectric('extract_dielectric')

        extract_dielectric.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        extract_dielectric.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_production.id)
        extract_dielectric.trajectory_path = ProtocolPath('trajectory_file_path', npt_production.id)
        extract_dielectric.system = ProtocolPath('system', assign_topology.id)

        # Set up a conditional group to ensure convergence of uncertainty
        converge_uncertainty = groups.ConditionalGroup('converge_uncertainty')
        converge_uncertainty.add_protocols(npt_production, extract_dielectric)

        condition = groups.ConditionalGroup.Condition()

        condition.left_hand_value = ProtocolPath('value.uncertainty',
                                                 converge_uncertainty.id,
                                                 extract_dielectric.id)

        condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

        condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

        converge_uncertainty.add_condition(condition)

        converge_uncertainty.max_iterations = 1

        schema.protocols[converge_uncertainty.id] = converge_uncertainty.schema

        # Finally, extract uncorrelated data
        extract_uncorrelated_trajectory = protocols.ExtractUncorrelatedTrajectoryData('extract_traj')

        extract_uncorrelated_trajectory.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                                converge_uncertainty.id,
                                                                                extract_dielectric.id)

        extract_uncorrelated_trajectory.equilibration_index = ProtocolPath('equilibration_index',
                                                                           converge_uncertainty.id,
                                                                           extract_dielectric.id)

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
                                                                                extract_dielectric.id)

        extract_uncorrelated_statistics.equilibration_index = ProtocolPath('equilibration_index',
                                                                           converge_uncertainty.id,
                                                                           extract_dielectric.id)

        extract_uncorrelated_statistics.input_statistics_path = ProtocolPath('statistics_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        schema.protocols[extract_uncorrelated_statistics.id] = extract_uncorrelated_statistics.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath('value', converge_uncertainty.id, extract_dielectric.id)

        output_to_store = WorkflowOutputToStore()

        output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                            extract_uncorrelated_trajectory.id)
        output_to_store.coordinate_file_path = ProtocolPath('output_coordinate_file',
                                                            converge_uncertainty.id, npt_production.id)

        output_to_store.statistics_file_path = ProtocolPath('output_statistics_path',
                                                            extract_uncorrelated_statistics.id)

        output_to_store.statistical_inefficiency = ProtocolPath('statistical_inefficiency', converge_uncertainty.id,
                                                                extract_dielectric.id)

        schema.outputs_to_store = {'full_system': output_to_store}

        return schema

    @staticmethod
    def get_default_reweighting_workflow_schema():

        schema = WorkflowSchema(property_type=DielectricConstant.__name__)
        schema.id = '{}{}'.format(DielectricConstant.__name__, 'Schema')

        # Unpack all the of the stored data.
        unpack_stored_data = protocols.UnpackStoredSimulationData('unpack_data_$(data_repl)')
        unpack_stored_data.simulation_data_path = ReplicatorValue('data_repl')

        schema.protocols[unpack_stored_data.id] = unpack_stored_data.schema

        # Calculate the autocorrelation time of each of the stored files for this property.
        build_reference_system = protocols.BuildSmirnoffSystem('build_system_$(data_repl)')

        build_reference_system.force_field_path = ProtocolPath('force_field_path', unpack_stored_data.id)
        build_reference_system.substance = ProtocolPath('substance', unpack_stored_data.id)
        build_reference_system.coordinate_file_path = ProtocolPath('coordinate_file_path',
                                                                   unpack_stored_data.id)

        dielectric_calculation = ExtractAverageDielectric('calc_dielectric_$(data_repl)')

        dielectric_calculation.thermodynamic_state = ProtocolPath('thermodynamic_state',
                                                                  unpack_stored_data.id)
        dielectric_calculation.input_coordinate_file = ProtocolPath('coordinate_file_path',
                                                                    unpack_stored_data.id)
        dielectric_calculation.trajectory_path = ProtocolPath('trajectory_file_path',
                                                              unpack_stored_data.id)
        dielectric_calculation.system = ProtocolPath('system', build_reference_system.id)

        schema.protocols[dielectric_calculation.id] = dielectric_calculation.schema

        # Decorrelate the frames of the concatenated trajectory.
        decorrelate_trajectory = protocols.ExtractUncorrelatedTrajectoryData('decorrelate_traj_$(data_repl)')

        decorrelate_trajectory.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                       dielectric_calculation.id)
        decorrelate_trajectory.equilibration_index = ProtocolPath('equilibration_index',
                                                                  dielectric_calculation.id)
        decorrelate_trajectory.input_coordinate_file = ProtocolPath('coordinate_file_path',
                                                                    unpack_stored_data.id)
        decorrelate_trajectory.input_trajectory_path = ProtocolPath('trajectory_file_path',
                                                                    unpack_stored_data.id)

        schema.protocols[decorrelate_trajectory.id] = decorrelate_trajectory.schema

        # Stitch together all of the trajectories
        concatenate_trajectories = protocols.ConcatenateTrajectories('concat_traj')

        concatenate_trajectories.input_coordinate_paths = [ProtocolPath('coordinate_file_path',
                                                                        unpack_stored_data.id)]

        concatenate_trajectories.input_trajectory_paths = [ProtocolPath('output_trajectory_path',
                                                                        decorrelate_trajectory.id)]

        schema.protocols[concatenate_trajectories.id] = concatenate_trajectories.schema

        # Calculate the reduced potentials for each of the reference states.
        schema.protocols[build_reference_system.id] = build_reference_system.schema

        reduced_reference_potential = protocols.CalculateReducedPotentialOpenMM('reduced_potential_$(data_repl)')

        reduced_reference_potential.system = ProtocolPath('system', build_reference_system.id)
        reduced_reference_potential.thermodynamic_state = ProtocolPath('thermodynamic_state',
                                                                       unpack_stored_data.id)
        reduced_reference_potential.coordinate_file_path = ProtocolPath('coordinate_file_path',
                                                                        unpack_stored_data.id)
        reduced_reference_potential.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                        concatenate_trajectories.id)

        schema.protocols[reduced_reference_potential.id] = reduced_reference_potential.schema

        # Calculate the reduced potential of the target state.
        build_target_system = protocols.BuildSmirnoffSystem('build_system_target')

        build_target_system.force_field_path = ProtocolPath('force_field_path', 'global')
        build_target_system.substance = ProtocolPath('substance', 'global')
        build_target_system.coordinate_file_path = ProtocolPath('output_coordinate_path',
                                                                concatenate_trajectories.id)

        schema.protocols[build_target_system.id] = build_target_system.schema

        reduced_target_potential = protocols.CalculateReducedPotentialOpenMM('reduced_potential_target')

        reduced_target_potential.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
        reduced_target_potential.system = ProtocolPath('system', build_target_system.id)
        reduced_target_potential.coordinate_file_path = ProtocolPath('output_coordinate_path',
                                                                     concatenate_trajectories.id)
        reduced_target_potential.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                     concatenate_trajectories.id)

        schema.protocols[reduced_target_potential.id] = reduced_target_potential.schema

        # Finally, apply MBAR to get the reweighted value.
        mbar_protocol = protocols.ReweightWithMBARProtocol('mbar')

        mbar_protocol.reference_reduced_potentials = [ProtocolPath('reduced_potentials',
                                                                   reduced_reference_potential.id)]

        mbar_protocol.reference_observables = [ProtocolPath('uncorrelated_values', dielectric_calculation.id)]
        mbar_protocol.target_reduced_potentials = [ProtocolPath('reduced_potentials', reduced_target_potential.id)]

        schema.protocols[mbar_protocol.id] = mbar_protocol.schema

        # Create the replicator object.
        component_replicator = ProtocolReplicator(replicator_id='data_repl')

        component_replicator.protocols_to_replicate = []

        # Pass it paths to the protocols to be replicated.
        for protocol in schema.protocols.values():

            if protocol.id.find('$(data_repl)') < 0:
                continue

            component_replicator.protocols_to_replicate.append(ProtocolPath('', protocol.id))

        component_replicator.template_values = ProtocolPath('full_system_data', 'global')

        schema.replicators = [component_replicator]

        schema.final_value_source = ProtocolPath('value', mbar_protocol.id)

        return schema