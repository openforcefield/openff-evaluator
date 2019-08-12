"""
A collection of dielectric physical property definitions.
"""

import logging

import numpy as np
from simtk import openmm

from propertyestimator import unit
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties import PhysicalProperty, PropertyPhase
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.protocols import analysis, reweighting
from propertyestimator.protocols.utils import generate_base_reweighting_protocols, BaseReweightingProtocols, \
    generate_gradient_protocol_group, generate_base_simulation_protocols
from propertyestimator.storage import StoredSimulationData
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import timeseries
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import bootstrap
from propertyestimator.workflow import plugins, WorkflowOptions
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@plugins.register_calculation_protocol()
class ExtractAverageDielectric(analysis.AverageTrajectoryProperty):
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
    def dipole_moments(self):
        """The raw (possibly correlated) dipole moments which were used in
        the dielectric calculation."""
        pass

    @protocol_output(unit.Quantity)
    def volumes(self):
        """The volumes which were used in the dielectric calculation."""
        pass

    @protocol_output(unit.Quantity)
    def uncorrelated_volumes(self):
        """The uncorrelated volumes which were used in the dielectric
        calculation."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._system_path = None
        self._system = None

        self._thermodynamic_state = None

        self._uncorrelated_volumes = None
        self._dipole_moments = None
        self._volumes = None

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
                                                       unit.boltzmann_constant *
                                                       temperature *
                                                       volume *
                                                       e0)

        return dielectric_constant

    def execute(self, directory, available_resources):

        import mdtraj
        from simtk import unit as simtk_unit
        from simtk.openmm import XmlSerializer

        logging.info('Extracting dielectrics: ' + self.id)

        base_exception = super(ExtractAverageDielectric, self).execute(directory, available_resources)

        if isinstance(base_exception, ExtractAverageDielectric):
            return base_exception

        charge_list = []

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        for force_index in range(self._system.getNumForces()):

            force = self._system.getForce(force_index)

            if not isinstance(force, openmm.NonbondedForce):
                continue

            for atom_index in range(force.getNumParticles()):

                charge = force.getParticleParameters(atom_index)[0]
                charge = charge.value_in_unit(simtk_unit.elementary_charge)

                charge_list.append(charge)

        dipole_moments = mdtraj.geometry.dipole_moments(self.trajectory, charge_list)
        self._dipole_moments = dipole_moments * unit.dimensionless

        dipole_moments, self._equilibration_index, self._statistical_inefficiency = \
            timeseries.decorrelate_time_series(dipole_moments)

        sample_indices = timeseries.get_uncorrelated_indices(len(self.trajectory[self._equilibration_index:]),
                                                             self._statistical_inefficiency)

        sample_indices = [index + self._equilibration_index for index in sample_indices]

        volumes = self.trajectory[sample_indices].unitcell_volumes
        self._volumes = self.trajectory.unitcell_volumes * unit.nanometer ** 3

        self._uncorrelated_values = dipole_moments * unit.dimensionless
        self._uncorrelated_volumes = volumes * unit.nanometer ** 3

        value, uncertainty = bootstrap(self._bootstrap_function,
                                       self._bootstrap_iterations,
                                       self._bootstrap_sample_size,
                                       dipoles=dipole_moments,
                                       volumes=volumes)

        self._value = EstimatedQuantity(value * unit.dimensionless,
                                        uncertainty * unit.dimensionless, self.id)

        logging.info('Extracted dielectrics: ' + self.id)

        return self._get_output_dictionary()


@plugins.register_calculation_protocol()
class ReweightDielectricConstant(reweighting.ReweightWithMBARProtocol):
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
                                                       unit.boltzmann_constant *
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
            error = self._execute_with_bootstrapping(unit.dimensionless,
                                                     dipoles=dipole_moments,
                                                     dipoles_sqr=dipole_moments_sqr,
                                                     volumes=volumes)
        else:

            return PropertyEstimatorException(directory=directory,
                                              message='Dielectric constant can only be reweighted in conjunction '
                                                      'with bootstrapped uncertainties.')

        if error is not None:

            error.directory = directory
            return error

        return self._get_output_dictionary()


@register_estimable_property()
@register_thermoml_property(thermoml_string='Relative permittivity at zero frequency',
                            supported_phases=PropertyPhase.Liquid)
class DielectricConstant(PhysicalProperty):
    """A class representation of a dielectric property"""

    @property
    def multi_component_property(self):
        return False

    @property
    def required_data_class(self):
        return StoredSimulationData

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
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Define the protocol which will extract the average dielectric constant
        # from the results of a simulation.
        extract_dielectric = ExtractAverageDielectric('extract_dielectric')
        extract_dielectric.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        # Define the protocols which will run the simulation itself.
        protocols, value_source, output_to_store = generate_base_simulation_protocols(extract_dielectric,
                                                                                      options)

        # Make sure the input of the analysis protcol is properly hooked up.
        extract_dielectric.system_path = ProtocolPath('system_path', protocols.assign_parameters.id)

        # Dielectric constants typically take longer to converge, so we need to
        # reflect this in the maximum number of convergence iterations.
        protocols.converge_uncertainty.max_iterations = 400

        # Set up the gradient calculations. For dielectric constants, we need to use
        # a slightly specialised reweighting protocol which we set up here.
        gradient_mbar_protocol = ReweightDielectricConstant('gradient_mbar')
        gradient_mbar_protocol.reference_observables = [ProtocolPath('dipole_moments',
                                                                     protocols.converge_uncertainty.id,
                                                                     extract_dielectric.id)]
        gradient_mbar_protocol.reference_volumes = [ProtocolPath('volumes',
                                                                 protocols.converge_uncertainty.id,
                                                                 extract_dielectric.id)]
        gradient_mbar_protocol.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        coordinate_source = ProtocolPath('output_coordinate_file', protocols.equilibration_simulation.id)
        trajectory_source = ProtocolPath('trajectory_file_path',
                                         protocols.converge_uncertainty.id,
                                         protocols.production_simulation.id)

        gradient_group, gradient_replicator, gradient_source = \
            generate_gradient_protocol_group([ProtocolPath('force_field_path', 'global')],
                                             ProtocolPath('force_field_path', 'global'),
                                             coordinate_source,
                                             trajectory_source,
                                             template_reweighting_schema=gradient_mbar_protocol.schema)

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=DielectricConstant.__name__)
        schema.id = '{}{}'.format(DielectricConstant.__name__, 'Schema')

        schema.protocols = {
            protocols.build_coordinates.id: protocols.build_coordinates.schema,
            protocols.assign_parameters.id: protocols.assign_parameters.schema,
            protocols.energy_minimisation.id: protocols.energy_minimisation.schema,
            protocols.equilibration_simulation.id: protocols.equilibration_simulation.schema,
            protocols.converge_uncertainty.id: protocols.converge_uncertainty.schema,
            protocols.extract_uncorrelated_trajectory.id: protocols.extract_uncorrelated_trajectory.schema,
            protocols.extract_uncorrelated_statistics.id: protocols.extract_uncorrelated_statistics.schema,
            gradient_group.id: gradient_group.schema
        }

        schema.replicators = [gradient_replicator]

        schema.outputs_to_store = {'full_system': output_to_store}

        schema.gradients_sources = [gradient_source]
        schema.final_value_source = value_source

        return schema

    @staticmethod
    def get_default_reweighting_workflow_schema(options=None):
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

        dielectric_calculation = ExtractAverageDielectric('calc_dielectric_$(data_repl)')
        base_reweighting_protocols, data_replicator = generate_base_reweighting_protocols(dielectric_calculation,
                                                                                          options)

        unpack_id = base_reweighting_protocols.unpack_stored_data.id

        # Set up a protocol to extract the dielectric constant from the stored data.
        dielectric_calculation.thermodynamic_state = ProtocolPath('thermodynamic_state', unpack_id)
        dielectric_calculation.input_coordinate_file = ProtocolPath('coordinate_file_path', unpack_id)
        dielectric_calculation.trajectory_path = ProtocolPath('trajectory_file_path', unpack_id)
        dielectric_calculation.system_path = ProtocolPath('system_path',
                                                          base_reweighting_protocols.build_reference_system.id)

        # For the dielectric constant, we employ a slightly more advanced reweighting
        # protocol set up for calculating fluctuation properties.
        mbar_protocol = ReweightDielectricConstant('mbar')

        mbar_protocol.reference_reduced_potentials = [ProtocolPath('statistics_file_path',
                                                                   base_reweighting_protocols.
                                                                   reduced_reference_potential.id)]

        mbar_protocol.reference_observables = [ProtocolPath('uncorrelated_values', dielectric_calculation.id)]
        mbar_protocol.reference_volumes = [ProtocolPath('uncorrelated_volumes', dielectric_calculation.id)]

        mbar_protocol.target_reduced_potentials = [ProtocolPath('statistics_file_path', base_reweighting_protocols.
                                                                                        reduced_target_potential.id)]

        mbar_protocol.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        mbar_protocol.bootstrap_uncertainties = True
        mbar_protocol.bootstrap_iterations = 200

        # TODO: Implement a cleaner way to handle this.
        if options.convergence_mode == WorkflowOptions.ConvergenceMode.NoChecks:
            mbar_protocol.required_effective_samples = -1

        # Make a copy of the mbar reweighting schema to use for evaulating gradients by reweighting.
        mbar_template_schema = mbar_protocol.schema

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
                                             template_reweighting_schema=mbar_template_schema,
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   mbar_protocol.id))

        schema = WorkflowSchema(property_type=DielectricConstant.__name__)
        schema.id = '{}{}'.format(DielectricConstant.__name__, 'Schema')

        schema.protocols = {protocol.id: protocol.schema for protocol in base_reweighting_protocols}
        schema.protocols[gradient_group.id] = gradient_group.schema

        schema.replicators = [data_replicator, gradient_replicator]

        schema.gradients_sources = [gradient_source]
        schema.final_value_source = ProtocolPath('value', base_reweighting_protocols.mbar_protocol.id)

        return schema
