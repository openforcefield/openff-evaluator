"""
A collection of dielectric physical property definitions.
"""

import logging
import sys

import numpy as np
from simtk import openmm, unit

from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.properties.properties import PhysicalProperty
from propertyestimator.properties.utils import generate_base_reweighting_protocols, BaseReweightingProtocols
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils import timeseries
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import bootstrap
from propertyestimator.workflow import protocols, groups, plugins
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.schemas import WorkflowOutputToStore, WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@plugins.register_calculation_protocol()
class ExtractAverageDielectric(protocols.AverageTrajectoryProperty):
    """Extracts the average dielectric constant from a simulation trajectory.
    """

    @protocol_input(str)
    def system_path(self):
        """The path to the XML system object which defines the forces present in the system."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state at which the trajectory was generated."""
        pass

    @protocol_output(unit.Quantity)
    def uncorrelated_volumes(self):
        """The uncorrelated volumes which were used in the dielect
        calculation."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._system_path = None
        self._system = None

        self._thermodynamic_state = None

        self._uncorrelated_volumes = None

    def _bootstrap_function(self, **sample_kwargs):
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
        sample_kwargs: dict of str and np.ndarray
            A key words dictionary of the bootstrap sample data, where the
            sample data is a numpy array of shape=(num_frames, num_dimensions)
            with dtype=float. The kwargs should include the dipole moment and
            the system volume

        Returns
        -------
        float
            The unitless static dielectric constant
        """

        dipole_moments = sample_kwargs['dipoles']
        volumes = sample_kwargs['volumes']

        temperature = self._thermodynamic_state.temperature

        dipole_mu = dipole_moments.mean(0)
        shifted_dipoles = dipole_moments - dipole_mu

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

        import mdtraj

        logging.info('Extracting dielectrics: ' + self.id)

        base_exception = super(ExtractAverageDielectric, self).execute(directory, available_resources)

        if isinstance(base_exception, ExtractAverageDielectric):
            return base_exception

        charge_list = []

        from simtk.openmm import XmlSerializer

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        for force_index in range(self._system.getNumForces()):

            force = self._system.getForce(force_index)

            if not isinstance(force, openmm.NonbondedForce):
                continue

            for atom_index in range(force.getNumParticles()):

                charge = force.getParticleParameters(atom_index)[0]
                charge /= unit.elementary_charge

                charge_list.append(charge)

        dipole_moments = mdtraj.geometry.dipole_moments(self.trajectory, charge_list)

        dipole_moments, self._equilibration_index, self._statistical_inefficiency = \
            timeseries.decorrelate_time_series(dipole_moments)

        sample_indices = timeseries.get_uncorrelated_indices(len(self.trajectory[self._equilibration_index:]),
                                                             self._statistical_inefficiency)

        sample_indices = [index + self._equilibration_index for index in sample_indices]

        volumes = self.trajectory[sample_indices].unitcell_volumes

        self._uncorrelated_values = unit.Quantity(dipole_moments, None)
        self._uncorrelated_volumes = volumes * unit.nanometer ** 3

        value, uncertainty = bootstrap(self._bootstrap_function,
                                       self._bootstrap_iterations,
                                       self._bootstrap_sample_size,
                                       dipoles=dipole_moments,
                                       volumes=volumes)

        self._value = EstimatedQuantity(unit.Quantity(value, None),
                                        unit.Quantity(uncertainty, None), self.id)

        logging.info('Extracted dielectrics: ' + self.id)

        return self._get_output_dictionary()


@plugins.register_calculation_protocol()
class ReweightDielectricConstant(protocols.ReweightWithMBARProtocol):
    """Reweights a set of dipole moments (`reference_observables`) and volumes
    (`reference_volumes`) using MBAR, and then combines these to yeild the reweighted
    dielectric constant. Uncertainties in the dielectric constant are determined
    by bootstrapping.
    """
    @protocol_input(unit.Quantity)
    def reference_volumes(self):
        """The a Quantity wrapped np.ndarray of the volumes of each of the
        reference states."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state at which the trajectory was generated."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new ReweightFluctuationWithMBARProtocol object."""

        super().__init__(protocol_id)

        self._thermodynamic_state = None

        self._reference_volumes = None
        self._bootstrap_uncertainties = True

    def _bootstrap_function(self, reference_reduced_potentials, target_reduced_potentials, **reference_observables):

        assert len(reference_observables) == 3

        transposed_observables = {}

        for key in reference_observables:
            transposed_observables[key] = np.transpose(reference_observables[key])

        values, _, _ = self._reweight_observables(np.transpose(reference_reduced_potentials),
                                                  np.transpose(target_reduced_potentials),
                                                  **transposed_observables)

        average_squared_dipole = values['dipoles_sqr']
        average_dipole_squared = np.linalg.norm(values['dipoles'])

        dipole_variance = (average_squared_dipole - average_dipole_squared) * \
                          (unit.elementary_charge * unit.nanometers) ** 2

        volume = values['volumes'] * unit.nanometer ** 3

        e0 = 8.854187817E-12 * unit.farad / unit.meter  # Taken from QCElemental

        dielectric_constant = 1.0 + dipole_variance / (3 *
                                                       unit.BOLTZMANN_CONSTANT_kB *
                                                       self._thermodynamic_state.temperature *
                                                       volume *
                                                       e0)

        return dielectric_constant

    def execute(self, directory, available_resources):

        logging.info('Reweighting dielectric: {}'.format(self.id))

        if len(self._reference_observables) == 0:
            return PropertyEstimatorException(directory=directory, message='There were no dipole moments to reweight.')

        if len(self._reference_volumes) == 0:
            return PropertyEstimatorException(directory=directory, message='There were no volumes to reweight.')

        if (not isinstance(self._reference_observables[0], unit.Quantity) or
            not isinstance(self._reference_volumes[0], unit.Quantity)):

            return PropertyEstimatorException(directory=directory,
                                              message='The reference observables should be '
                                                      'a list of unit.Quantity wrapped ndarray\'s.')

        if len(self._reference_observables) != len(self._reference_volumes):
            return PropertyEstimatorException(directory=directory, message='The number of reference dipoles does '
                                                                           'not match the number of reference volumes.')

        for reference_dipoles, reference_volumes in zip(self._reference_observables, self._reference_volumes):

            if len(reference_dipoles) == len(reference_volumes):
                continue

            return PropertyEstimatorException(directory=directory, message='The number of reference dipoles does '
                                                                           'not match the number of reference volumes.')

        dipole_moments = self._prepare_observables_array(self._reference_observables)
        dipole_moments_sqr = np.array([[np.dot(dipole, dipole) for dipole in np.transpose(dipole_moments)]])

        volumes = self._prepare_observables_array(self._reference_volumes)

        if self._bootstrap_uncertainties:

            reference_potentials = np.transpose(np.array(self._reference_reduced_potentials))
            target_potentials = np.transpose(np.array(self._target_reduced_potentials))

            frame_counts = np.array([len(observable) for observable in self._reference_observables])

            # Construct an mbar object to get out the number of effective samples.
            import pymbar
            mbar = pymbar.MBAR(self._reference_reduced_potentials,
                               frame_counts, verbose=False, relative_tolerance=1e-12)

            effective_samples = mbar.computeEffectiveSampleNumber().max()

            value, uncertainty = bootstrap(self._bootstrap_function,
                                           self._bootstrap_iterations,
                                           self._bootstrap_sample_size,
                                           frame_counts,
                                           reference_reduced_potentials=reference_potentials,
                                           target_reduced_potentials=target_potentials,
                                           dipoles=np.transpose(dipole_moments),
                                           dipoles_sqr=np.transpose(dipole_moments_sqr),
                                           volumes=np.transpose(volumes))

            if effective_samples < self._required_effective_samples:
                uncertainty = sys.float_info.max

            self._value = EstimatedQuantity(unit.Quantity(value, None),
                                            unit.Quantity(uncertainty, None),
                                            self.id)
            
        else:

            return PropertyEstimatorException(directory=directory, message='Dielectric uncertainties may only'
                                                                           'be bootstrapped.')

        logging.info('Dielectric reweighted: {}'.format(self.id))

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
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return DielectricConstant.get_default_simulation_workflow_schema(options)
        elif calculation_layer == 'ReweightingLayer':
            return DielectricConstant.get_default_reweighting_workflow_schema(options)

        return None

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        from direct simulations.

        Parameters
        ----------
        options: PropertyWorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

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
        energy_minimisation.system_path = ProtocolPath('system_path', assign_topology.id)

        schema.protocols[energy_minimisation.id] = energy_minimisation.schema

        npt_equilibration = protocols.RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 100000  # Debug settings.
        npt_equilibration.output_frequency = 5000  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system_path = ProtocolPath('system_path', assign_topology.id)

        schema.protocols[npt_equilibration.id] = npt_equilibration.schema

        # Production
        npt_production = protocols.RunOpenMMSimulation('npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 500000  # Debug settings.
        npt_production.output_frequency = 10000  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system_path = ProtocolPath('system_path', assign_topology.id)

        # Analysis
        extract_dielectric = ExtractAverageDielectric('extract_dielectric')

        extract_dielectric.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        extract_dielectric.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_production.id)
        extract_dielectric.trajectory_path = ProtocolPath('trajectory_file_path', npt_production.id)
        extract_dielectric.system_path = ProtocolPath('system_path', assign_topology.id)

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

        converge_uncertainty.max_iterations = 400

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
    def get_default_reweighting_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        by reweighting existing data.

        Parameters
        ----------
        options: PropertyWorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        dielectric_calculation = ExtractAverageDielectric('calc_dielectric_$(data_repl)')
        base_reweighting_protocols, data_replicator = generate_base_reweighting_protocols(dielectric_calculation)

        unpack_id = base_reweighting_protocols.unpack_stored_data.id

        dielectric_calculation.thermodynamic_state = ProtocolPath('thermodynamic_state', unpack_id)
        dielectric_calculation.input_coordinate_file = ProtocolPath('coordinate_file_path', unpack_id)
        dielectric_calculation.trajectory_path = ProtocolPath('trajectory_file_path', unpack_id)
        dielectric_calculation.system_path = ProtocolPath('system_path', base_reweighting_protocols.build_reference_system.id)

        # For the dielectric constant, we employ a slightly more advanced protocol
        # set up for calculating fluctuation properties.
        mbar_protocol = ReweightDielectricConstant('mbar')

        mbar_protocol.reference_reduced_potentials = [ProtocolPath('reduced_potentials',
                                                                   base_reweighting_protocols.
                                                                   reduced_reference_potential.id)]

        mbar_protocol.reference_observables = [ProtocolPath('uncorrelated_values', dielectric_calculation.id)]
        mbar_protocol.reference_volumes = [ProtocolPath('uncorrelated_volumes', dielectric_calculation.id)]

        mbar_protocol.target_reduced_potentials = [ProtocolPath('reduced_potentials', base_reweighting_protocols.
                                                                                      reduced_target_potential.id)]

        mbar_protocol.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        mbar_protocol.bootstrap_uncertainties = True
        mbar_protocol.bootstrap_iterations = 200

        # Recreate the immutable tuple for convenience.
        base_reweighting_protocols = BaseReweightingProtocols(base_reweighting_protocols.unpack_stored_data,
                                                              base_reweighting_protocols.analysis_protocol,
                                                              base_reweighting_protocols.decorrelate_trajectory,
                                                              base_reweighting_protocols.concatenate_trajectories,
                                                              base_reweighting_protocols.build_reference_system,
                                                              base_reweighting_protocols.reduced_reference_potential,
                                                              base_reweighting_protocols.build_target_system,
                                                              base_reweighting_protocols.reduced_target_potential,
                                                              mbar_protocol)

        schema = WorkflowSchema(property_type=DielectricConstant.__name__)
        schema.id = '{}{}'.format(DielectricConstant.__name__, 'Schema')

        schema.protocols = {protocol.id: protocol.schema for protocol in base_reweighting_protocols}
        schema.replicators = [data_replicator]

        schema.final_value_source = ProtocolPath('value', base_reweighting_protocols.mbar_protocol.id)

        return schema
