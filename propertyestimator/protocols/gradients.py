"""
A collection of protocols for reweighting cached simulation data.
"""
import copy

import numpy as np
from simtk import unit, openmm
from simtk.openmm import app

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
    def reference_reduced_potentials(self):
        """The estimated gradients."""
        pass

    @protocol_output(float)
    def reverse_parameter_value(self):
        pass

    @protocol_output(float)
    def forward_parameter_value(self):
        pass

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

        self._reverse_parameter_value = None
        self._forward_parameter_value = None

        self._reference_reduced_potentials = None
        self._reverse_reduced_potentials = None
        self._forward_reduced_potentials = None

    def _build_reduced_system(self, original_force_field, topology, scale_amount=None):
        """Produces an OpenMM system containing only forces for the specified parameter,
         optionally perturbed by the amount specified by `scale_amount`.

        Parameters
        ----------
        original_force_field: openforcefield.typing.engines.smirnoff.ForceField
            The force field to create the system from (and optionally perturb).
        topology: openforcefield.topology.Topology
            The topology of the system to apply the force field to.
        scale_amount: float, optional
            The optional amount to perturb the parameter by.

        Returns
        -------
        simtk.openmm.System
            The created system.
        float
            The new value of the perturbed parameter.
        """
        from openforcefield.typing.engines.smirnoff import ForceField

        parameter_tag = self._parameter_tuple.tag
        parameter_smirks = self._parameter_tuple.smirks
        parameter_attribute = self._parameter_tuple.attribute

        original_handler = original_force_field.get_parameter_handler(parameter_tag)
        original_parameter = original_handler.parameters[parameter_smirks]

        if self._use_subset_of_force_field:

            force_field = ForceField()
            handler = copy.deepcopy(original_force_field.get_parameter_handler(parameter_tag))
            force_field.register_parameter_handler(handler)

        else:

            force_field = copy.deepcopy(original_force_field)
            handler = force_field.get_parameter_handler(parameter_tag)

        parameter_value = getattr(original_parameter, parameter_attribute)

        if scale_amount is not None:

            existing_parameter = handler.parameters[parameter_smirks]

            parameter_value *= (1.0 + scale_amount)
            setattr(existing_parameter, parameter_attribute, parameter_value)

        system = force_field.create_openmm_system(topology,
                                                  allow_missing_parameters=True)

        return system, parameter_value

    def _evaluate_reduced_potential(self, system, trajectory, compute_resources):
        """Return the potential energy.
        Parameters
        ----------
        system: simtk.openmm.System
            The system which encodes the interaction forces for the
            specified parameter.
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

        from openforcefield.topology import Molecule, Topology
        from openforcefield.typing.engines.smirnoff import ForceField

        original_force_field = ForceField(self._force_field_path)

        trajectory = mdtraj.load_dcd(self._trajectory_file_path,
                                     self._coordinate_file_path)

        unique_molecules = []

        for component in self._substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)
            unique_molecules.append(molecule)

        pdb_file = app.PDBFile(self._coordinate_file_path)
        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)

        # Build the reduced reference force field
        reference_system, _ = self._build_reduced_system(original_force_field, topology)

        # Build the slightly perturbed force fields.
        reverse_system, self._reverse_parameter_value = self._build_reduced_system(original_force_field,
                                                                                   topology,
                                                                                   -self._perturbation_scale)

        forward_system, self._forward_parameter_value = self._build_reduced_system(original_force_field,
                                                                                   topology,
                                                                                   self._perturbation_scale)

        # Calculate the reduced potentials.
        self._reference_reduced_potentials, error = self._evaluate_reduced_potential(reference_system,
                                                                                     trajectory,
                                                                                     available_resources)

        if isinstance(error, PropertyEstimatorException):
            return error

        self._reverse_reduced_potentials, error = self._evaluate_reduced_potential(reverse_system,
                                                                                   trajectory,
                                                                                   available_resources)

        if isinstance(error, PropertyEstimatorException):
            return error

        self._forward_reduced_potentials, error = self._evaluate_reduced_potential(forward_system,
                                                                                   trajectory,
                                                                                   available_resources)

        if isinstance(error, PropertyEstimatorException):
            return error

        return self._get_output_dictionary()
