"""
A collection of protocols for reweighting cached simulation data.
"""
import copy
from os import path

import numpy as np
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import unit, openmm
from simtk.openmm import XmlSerializer

from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class GradientReducedPotentials(BaseProtocol):
    """A protocol to estimates the gradient of an observable with
    respect to a number of specified force field parameters.
    """

    @protocol_input(str)
    def force_field_path(self):
        """A path to the force field which contains the parameters
        to differentiate the observable with respect to."""
        pass

    @protocol_output(Substance)
    def substance(self):
        """The substance which describes the composition
        of the system."""
        pass

    @protocol_output(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state to estimate the gradients at."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """A path to the initial coordinates of the simulation trajectory which
        was used to estimate the observable of interest."""
        pass

    @protocol_output(str)
    def trajectory_file_path(self):
        """A path to the simulation trajectory which was used
        to estimate the observable of interest."""
        pass

    @protocol_input(tuple)
    def parameter_tuple(self):
        """A list of the parameters to differentiate with respect to."""
        pass

    @protocol_input(float)
    def perturbation_scale(self):
        """The amount to perturb the parameter by, such that
        p_new = p_old * (1 +/- perturbation_scale)"""
        pass

    @protocol_input(bool)
    def use_subset_of_force_field(self):
        """If true, the reduced potential will be estimated using an OpenMM
        system which only contains the parameter of interest.
        """

    @protocol_output(np.ndarray)
    def reverse_reduced_potentials(self):
        """The estimated gradients."""
        pass

    @protocol_output(np.ndarray)
    def forward_reduced_potentials(self):
        """The estimated gradients."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new EstimateParameterGradients object."""
        super().__init__(protocol_id)

        self._force_field_path = None

        self._substance = None
        self._thermodynamic_state = None

        self._statistical_inefficiency = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._parameter_tuple = None
        self._perturbation_scale = 1.0e-4

        self._use_subset_of_force_field = True

        self._reverse_reduced_potentials = None
        self._forward_reduced_potentials = None

    def _build_perturbed_force_field(self, original_force_field, output_path, scale_amount):
        """Produces a force field containing the specified parameter perturbed
        by the amount specified by `scale_amount`.

        Parameters
        ----------
        output_path: str
            The path to save the force field to.
        scale_amount: float
            The amount to perturb the parameter by.
        """

        parameter_tag = self._parameter_tuple.parameter_tag
        parameter_smirks = self._parameter_tuple.parameter_smirks
        parameter_attribute = self._parameter_tuple.parameter_attribute

        original_handler = original_force_field.get_parameter_handler(parameter_tag)
        original_parameter = original_handler[parameter_smirks]

        if self._use_subset_of_force_field:

            force_field = ForceField()
            handler = copy.deepcopy(original_force_field.get_parameter_handler(parameter_tag))
            force_field.register_parameter_handler(handler)

        else:

            force_field = copy.deepcopy(original_force_field)
            handler = force_field.get_parameter_handler(parameter_tag)

        existing_parameter = handler.parameters[parameter_smirks]

        reverse_value = getattr(original_parameter, parameter_attribute) * (1.0 + scale_amount)
        setattr(existing_parameter, parameter_attribute, reverse_value)

        force_field.to_file(output_path, 'XML')

    def _evaluate_reduced_potential(self, directory, force_field_path, trajectory, compute_resources):
        """Return the potential energy.
        Parameters
        ----------
        directory: str
            The directory that this protocol is executing in.
        force_field_path: str
            The path to the force field to use when evaluating the energy.
        trajectory: mdtraj.Trajectory
            A trajectory of configurations to evaluate.
        compute_resources: ComputeResources
            The compute resources available to execute on.

        Returns
        ---------
        simtk.unit.Quantity
            A unit bearing `np.ndarray` which contains the reduced potential.
        PropertyEstimatorException, optional
            Any exceptions that were raised.
        """

        build_system = BuildSmirnoffSystem('build_system')
        build_system.allow_missing_parameters = True
        build_system.apply_known_charges = False
        build_system.force_field_path = force_field_path
        build_system.substance = self._substance
        build_system.coordinate_file_path = self._coordinate_file_path

        build_system_result = build_system.execute(directory, compute_resources)

        if isinstance(build_system_result, PropertyEstimatorException):
            return None, build_system_result

        with open(build_system.system_path, 'r') as file:
            system = XmlSerializer.deserialize(file.read())

        integrator = openmm.VerletIntegrator(0.1 * unit.femtoseconds)

        platform = setup_platform_with_resources(compute_resources, True)
        openmm_context = openmm.Context(system, integrator, platform)

        reduced_potentials = np.zeros(trajectory.n_frames, dtype=np.float64)

        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * self._thermodynamic_state.temperature)

        for frame_index in range(trajectory.n_frames):

            positions = trajectory.xyz[frame_index]
            box_vectors = trajectory.openmm_boxes(frame_index)

            openmm_context.setPeriodicBoxVectors(*box_vectors)
            openmm_context.setPositions(positions)

            state = openmm_context.getState(getEnergy=True)

            unreduced_potential = state.getPotentialEnergy() / unit.AVOGADRO_CONSTANT_NA

            if self._thermodynamic_state.pressure is not None:
                unreduced_potential += self._thermodynamic_state.pressure * state.getPeriodicBoxVolume()

            # set box vectors
            reduced_potentials[frame_index] = unreduced_potential * beta

        reduced_potentials *= unit.dimensionless
        return reduced_potentials, None

    def execute(self, directory, available_resources):

        import mdtraj

        original_force_field = ForceField(self._force_field_path)

        reverse_path = path.join(directory, 'reverse.offxml')
        forward_path = path.join(directory, 'forward.offxml')

        trajectory = mdtraj.load_dcd(self._trajectory_file_path,
                                     self._coordinate_file_path)

        # Build the slightly perturbed force fields.
        self._build_perturbed_force_field(original_force_field,
                                          reverse_path,
                                          -self._perturbation_scale)

        self._build_perturbed_force_field(original_force_field,
                                          forward_path,
                                          self._perturbation_scale)

        # Calculate the reduced potentials..
        self._reverse_reduced_potentials, error = self._evaluate_reduced_potential(directory,
                                                                                   reverse_path,
                                                                                   trajectory,
                                                                                   available_resources)

        if isinstance(error, PropertyEstimatorException):
            return error

        self._forward_reduced_potentials, error = self._evaluate_reduced_potential(directory,
                                                                                   forward_path,
                                                                                   trajectory,
                                                                                   available_resources)

        if isinstance(error, PropertyEstimatorException):
            return error

        return self._get_output_dictionary()
