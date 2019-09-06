"""
A collection of protocols for reweighting cached simulation data.
"""
import copy
import logging
import re
from os import path

import numpy as np
from simtk import openmm
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.properties.properties import ParameterGradientKey, ParameterGradient
from propertyestimator.protocols.miscellaneous import BaseWeightByMoleFraction
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import pint_quantity_to_openmm, setup_platform_with_resources, \
    openmm_quantity_to_pint, disable_pbc
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class GradientReducedPotentials(BaseProtocol):
    """A protocol to estimates the gradient of an observable with
    respect to a number of specified force field parameters.
    """

    @protocol_input(list)
    def reference_force_field_paths(self):
        """A list of path to the force field file which were originally used
        to estimate the observable of interest."""
        pass

    @protocol_input(str)
    def reference_statistics_path(self):
        """An optional path to the statistics array which was generated
        alongside the observable of interest, which will be used to
        correct the potential energies at the reverse and forward states.

        This is only really needed when the observable of interest is an
        energy.
        """
        pass

    @protocol_input(str)
    def force_field_path(self):
        """A path to the force field which contains the parameters
        to differentiate the observable with respect to."""
        pass

    @protocol_input(bool)
    def enable_pbc(self):
        """If true, periodic boundary conditions will be enabled when
        re-evaluating the reduced potentials."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The substance which describes the composition
        of the system."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state to estimate the gradients at."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self):
        """A path to the initial coordinates of the simulation trajectory which
        was used to estimate the observable of interest."""
        pass

    @protocol_input(str)
    def trajectory_file_path(self):
        """A path to the simulation trajectory which was used
        to estimate the observable of interest."""
        pass

    @protocol_input(ParameterGradientKey)
    def parameter_key(self):
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

    @protocol_input(list)
    def effective_sample_indices(self):
        """NOTE - this is currently a placeholder input ONLY, and
        currently is not used for anything.
        """

    @protocol_output(list)
    def reference_potential_paths(self):
        pass

    @protocol_output(str)
    def reverse_potentials_path(self):
        pass

    @protocol_output(str)
    def forward_potentials_path(self):
        pass

    @protocol_output(unit.Quantity)
    def reverse_parameter_value(self):
        pass

    @protocol_output(unit.Quantity)
    def forward_parameter_value(self):
        pass

    def __init__(self, protocol_id):
        """Constructs a new EstimateParameterGradients object."""
        super().__init__(protocol_id)

        self._reference_force_field_paths = None
        self._reference_statistics_path = ''

        self._force_field_path = None
        self._enable_pbc = True

        self._substance = None
        self._thermodynamic_state = None

        self._statistical_inefficiency = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._parameter_key = None
        self._perturbation_scale = 1.0e-4

        self._use_subset_of_force_field = True

        self._reference_potential_paths = []
        self._reverse_potentials_path = None
        self._forward_potentials_path = None

        self._reverse_parameter_value = None
        self._forward_parameter_value = None

        # This is currently a placeholder variable ONLY
        # and should not be used for anything.
        self._effective_sample_indices = []

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
        simtk.unit.Quantity
            The new value of the perturbed parameter.
        """
        # As this method deals mainly with the toolkit, we stick to
        # simtk units here.
        from openforcefield.typing.engines.smirnoff import ForceField

        parameter_tag = self._parameter_key.tag
        parameter_smirks = self._parameter_key.smirks
        parameter_attribute = self._parameter_key.attribute

        original_handler = original_force_field.get_parameter_handler(parameter_tag)
        original_parameter = original_handler.parameters[parameter_smirks]

        if self._use_subset_of_force_field:

            force_field = ForceField()
            handler = copy.deepcopy(original_force_field.get_parameter_handler(parameter_tag))
            force_field.register_parameter_handler(handler)

        else:

            force_field = copy.deepcopy(original_force_field)
            handler = force_field.get_parameter_handler(parameter_tag)

        parameter_index = None
        value_list = None

        if hasattr(original_parameter, parameter_attribute):
            parameter_value = getattr(original_parameter, parameter_attribute)
        else:
            attribute_split = re.split(r'(\d+)', parameter_attribute)

            assert len(parameter_attribute) == 2
            assert hasattr(original_parameter, attribute_split[0])

            parameter_attribute = attribute_split[0]
            parameter_index = int(attribute_split[1]) - 1

            value_list = getattr(original_parameter, parameter_attribute)
            parameter_value = value_list[parameter_index]

        if scale_amount is not None:

            existing_parameter = handler.parameters[parameter_smirks]

            if np.isclose(parameter_value.value_in_unit(parameter_value.unit), 0.0):
                # Careful thought needs to be given to this. Consider cases such as
                # epsilon or sigma where negative values are not allowed.
                parameter_value = (scale_amount if scale_amount > 0.0 else 0.0) * parameter_value.unit
            else:
                parameter_value *= (1.0 + scale_amount)

            if value_list is None:
                setattr(existing_parameter, parameter_attribute, parameter_value)
            else:
                value_list[parameter_index] = parameter_value
                setattr(existing_parameter, parameter_attribute, value_list)

        system = force_field.create_openmm_system(topology)

        if not self._enable_pbc:
            disable_pbc(system)

        return system, parameter_value

    def _evaluate_reduced_potential(self, system, trajectory, file_path,
                                    compute_resources, subset_energy_corrections=None):
        """Return the potential energy.
        Parameters
        ----------
        system: simtk.openmm.System
            The system which encodes the interaction forces for the
            specified parameter.
        trajectory: mdtraj.Trajectory
            A trajectory of configurations to evaluate.
        file_path: str
            The path to save the reduced potentials to.
        compute_resources: ComputeResources
            The compute resources available to execute on.
        subset_energy_corrections: unit.Quantity, optional
            A unit.Quantity wrapped numpy.ndarray which contains a set
            of energies to add to the re-evaluated potential energies.
            This is mainly used to correct the potential energies evaluated
            using a subset of the force field back to energies as if evaluated
            using the full thing.

        Returns
        ---------
        propertyestimator.unit.Quantity
            A unit bearing `np.ndarray` which contains the reduced potential.
        PropertyEstimatorException, optional
            Any exceptions that were raised.
        """
        from simtk import unit as simtk_unit

        integrator = openmm.VerletIntegrator(0.1 * simtk_unit.femtoseconds)

        platform = setup_platform_with_resources(compute_resources, True)
        openmm_context = openmm.Context(system, integrator, platform)

        potentials = np.zeros(trajectory.n_frames, dtype=np.float64)
        reduced_potentials = np.zeros(trajectory.n_frames, dtype=np.float64)

        temperature = pint_quantity_to_openmm(self._thermodynamic_state.temperature)
        beta = 1.0 / (simtk_unit.BOLTZMANN_CONSTANT_kB * temperature)

        pressure = pint_quantity_to_openmm(self._thermodynamic_state.pressure)

        for frame_index in range(trajectory.n_frames):

            positions = trajectory.xyz[frame_index]
            box_vectors = trajectory.openmm_boxes(frame_index)

            if self._enable_pbc:
                openmm_context.setPeriodicBoxVectors(*box_vectors)

            openmm_context.setPositions(positions)

            state = openmm_context.getState(getEnergy=True)

            unreduced_potential = state.getPotentialEnergy() / simtk_unit.AVOGADRO_CONSTANT_NA

            if pressure is not None and self.enable_pbc:
                unreduced_potential += pressure * state.getPeriodicBoxVolume()

            potentials[frame_index] = state.getPotentialEnergy().value_in_unit(simtk_unit.kilojoule_per_mole)
            reduced_potentials[frame_index] = unreduced_potential * beta

        potentials *= unit.kilojoule / unit.mole
        reduced_potentials *= unit.dimensionless

        if subset_energy_corrections is not None:
            potentials += subset_energy_corrections

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials
        statistics_array[ObservableType.PotentialEnergy] = potentials
        statistics_array.to_pandas_csv(file_path)

    def execute(self, directory, available_resources):

        import mdtraj

        from openforcefield.topology import Molecule, Topology
        from openforcefield.typing.engines.smirnoff import ForceField

        logging.info(f'Calculating the reduced gradient potentials for {self._parameter_key}: {self._id}')

        if len(self._reference_force_field_paths) != 1 and self._use_subset_of_force_field:

            return PropertyEstimatorException(directory, 'A single reference force field must be '
                                                         'provided when calculating the reduced '
                                                         'potentials using a subset of the full force')

        if len(self._reference_statistics_path) <= 0 and self._use_subset_of_force_field:

            return PropertyEstimatorException(directory, 'The path to the statistics evaluated using '
                                                         'the full force field must be provided.')

        target_force_field = ForceField(self._force_field_path)

        trajectory = mdtraj.load_dcd(self._trajectory_file_path,
                                     self._coordinate_file_path)

        unique_molecules = []

        for component in self._substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)
            unique_molecules.append(molecule)

        pdb_file = app.PDBFile(self._coordinate_file_path)
        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)

        # If we are using only a subset of the system object, load in the reference
        # statistics containing the full system energies to correct the output
        # forward and reverse potential energies.
        reference_statistics = None
        subset_energy_corrections = None

        if self._use_subset_of_force_field:
            reference_statistics = StatisticsArray.from_pandas_csv(self._reference_statistics_path)

        # Compute the reduced reference energy if any reference force field files
        # have been provided.
        for index, reference_force_field_path in enumerate(self._reference_force_field_paths):

            reference_force_field = ForceField(reference_force_field_path, allow_cosmetic_attributes=True)
            reference_system, _ = self._build_reduced_system(reference_force_field, topology)

            reference_potentials_path = path.join(directory, f'reference_{index}.csv')

            self._evaluate_reduced_potential(reference_system, trajectory,
                                             reference_potentials_path,
                                             available_resources)

            self._reference_potential_paths.append(reference_potentials_path)

            if reference_statistics is not None:

                subset_energies = StatisticsArray.from_pandas_csv(reference_potentials_path)
                subset_energy_corrections = (reference_statistics[ObservableType.PotentialEnergy] -
                                             subset_energies[ObservableType.PotentialEnergy])

                subset_energies[ObservableType.PotentialEnergy] = reference_statistics[ObservableType.PotentialEnergy]
                subset_energies.to_pandas_csv(reference_potentials_path)

        # Build the slightly perturbed system.
        reverse_system, self._reverse_parameter_value = self._build_reduced_system(target_force_field,
                                                                                   topology,
                                                                                   -self._perturbation_scale)

        forward_system, self._forward_parameter_value = self._build_reduced_system(target_force_field,
                                                                                   topology,
                                                                                   self._perturbation_scale)

        self._reverse_parameter_value = openmm_quantity_to_pint(self._reverse_parameter_value)
        self._forward_parameter_value = openmm_quantity_to_pint(self._forward_parameter_value)

        # Calculate the reduced potentials.
        self._reverse_potentials_path = path.join(directory, 'reverse.csv')
        self._forward_potentials_path = path.join(directory, 'forward.csv')

        self._evaluate_reduced_potential(reverse_system, trajectory, self._reverse_potentials_path,
                                         available_resources, subset_energy_corrections)
        self._evaluate_reduced_potential(forward_system, trajectory, self._forward_potentials_path,
                                         available_resources, subset_energy_corrections)

        logging.info(f'Finished calculating the reduced gradient potentials.')

        return self._get_output_dictionary()


@register_calculation_protocol()
class CentralDifferenceGradient(BaseProtocol):
    """A protocol which employs the central diference method
    to estimate the gradient of an observable A, such that

    grad = (A(x-h) - A(x+h)) / (2h)

    Notes
    -----
    The `values` input must either be a list of unit.Quantity, a ProtocolPath to a list
    of unit.Quantity, or a list of ProtocolPath which each point to a unit.Quantity.
    """

    @protocol_input(ParameterGradientKey)
    def parameter_key(self):
        """The key that describes which parameters this
        gradient was estimated for."""
        pass

    @protocol_input(EstimatedQuantity)
    def reverse_observable_value(self):
        """The value of A(x-h)."""
        pass

    @protocol_input(EstimatedQuantity)
    def forward_observable_value(self):
        """The value of A(x+h)."""
        pass

    @protocol_input(unit.Quantity)
    def reverse_parameter_value(self):
        """The value of x-h."""
        pass

    @protocol_input(unit.Quantity)
    def forward_parameter_value(self):
        """The value of x+h."""
        pass

    @protocol_output(ParameterGradient)
    def gradient(self):
        """The estimated gradient."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new CentralDifferenceGradient object."""
        super().__init__(protocol_id)

        self._parameter_key = None

        self._reverse_observable_value = None
        self._forward_observable_value = None

        self._reverse_parameter_value = None
        self._forward_parameter_value = None

        self._gradient = None

    def execute(self, directory, available_resources):

        if self._forward_parameter_value < self._reverse_parameter_value:

            return PropertyEstimatorException(f'The forward parameter value ({self._forward_parameter_value}) must '
                                              f'be larger than the reverse value ({self._reverse_parameter_value}).')

        gradient = ((self._forward_observable_value.value - self._reverse_observable_value.value) /
                    (self._forward_parameter_value - self._reverse_parameter_value))

        self._gradient = ParameterGradient(self._parameter_key, gradient)

        return self._get_output_dictionary()


@register_calculation_protocol()
class DivideGradientByScalar(BaseProtocol):
    """A protocol which divides a gradient by a specified scalar

    Notes
    -----
    Once a more robust type system is built-in, this will be deprecated
    by `DivideValue`.
    """

    @protocol_input(ParameterGradient)
    def value(self):
        """The value to divide."""
        pass

    @protocol_input(object)
    def divisor(self):
        """The scalar to divide by."""
        pass

    @protocol_output(ParameterGradient)
    def result(self):
        """The result of the division."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new DivideValue object."""
        super().__init__(protocol_id)

        self._value = None
        self._divisor = None

        self._result = None

    def execute(self, directory, available_resources):

        self._result = ParameterGradient(self._value.key,
                                         self._value.value / self._divisor)

        return self._get_output_dictionary()


@register_calculation_protocol()
class MultiplyGradientByScalar(BaseProtocol):
    """A protocol which multiplies a gradient by a specified scalar

    Notes
    -----
    Once a more robust type system is built-in, this will be deprecated
    by `MultiplyValue`.
    """

    @protocol_input(ParameterGradient)
    def value(self):
        """The value to divide."""
        pass

    @protocol_input(unit.Quantity)
    def scalar(self):
        """The scalar to multiply by."""
        pass

    @protocol_output(ParameterGradient)
    def result(self):
        """The result of the division."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new DivideValue object."""
        super().__init__(protocol_id)

        self._value = None
        self._scalar = None

        self._result = None

    def execute(self, directory, available_resources):

        self._result = ParameterGradient(self._value.key,
                                         self._value.value * self._scalar)

        return self._get_output_dictionary()


@register_calculation_protocol()
class AddGradients(BaseProtocol):
    """A temporary protocol to add together multiple gradients.

    Notes
    -----
    Once a more robust type system is built-in, this will be deprecated
    by `AddValues`.
    """

    @protocol_input(list)
    def values(self):
        """The gradients to add together."""
        pass

    @protocol_output(ParameterGradient)
    def result(self):
        """The sum of the values."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddGradients object."""
        super().__init__(protocol_id)

        self._values = None
        self._result = None

    def execute(self, directory, available_resources):

        if len(self._values) < 1:
            return PropertyEstimatorException(directory, 'There were no gradients to add together')

        gradient_key = self._values[0].key
        gradient_value = None

        for gradient in self._values:

            if gradient_key == gradient.key:
                continue

            return PropertyEstimatorException(directory,
                                              f'Only gradients with the same key can be '
                                              f'added together (a={gradient_key} b={gradient.key})')

        for gradient in self._values:

            if gradient_value is None:

                gradient_value = gradient.value
                continue

            gradient_value += gradient.value

        self._result = ParameterGradient(key=gradient_key,
                                         value=gradient_value)

        return self._get_output_dictionary()


@register_calculation_protocol()
class SubtractGradients(BaseProtocol):
    """A temporary protocol to add together two gradients.

    Notes
    -----
    Once a more robust type system is built-in, this will be deprecated
    by `SubtractValues`.
    """

    @protocol_input(ParameterGradient)
    def value_a(self):
        """`value_a` in the formula `result = value_b - value_a`"""
        pass

    @protocol_input(ParameterGradient)
    def value_b(self):
        """`value_b` in the formula  `result = value_b - value_a`"""
        pass

    @protocol_output(ParameterGradient)
    def result(self):
        """The sum of the values."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._value_a = None
        self._value_b = None

        self._result = None

    def execute(self, directory, available_resources):

        if self._value_a.key != self._value_b.key:

            return PropertyEstimatorException(directory=directory,
                                              message=f'Only gradients with the same key can be '
                                                      f'added together (a={self._value_a.key} b={self._value_b.key})')

        self._result = ParameterGradient(key=self._value_a.key,
                                         value=self._value_b.value - self._value_a.value)

        return self._get_output_dictionary()


@register_calculation_protocol()
class WeightGradientByMoleFraction(BaseWeightByMoleFraction):
    """Multiplies a gradient by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(ParameterGradient)
    def value(self):
        """The value to be weighted."""
        pass

    @protocol_output(ParameterGradient)
    def weighted_value(self, value):
        """The value weighted by the `component`s mole fraction as determined from
        the `full_substance`."""
        pass

    def _weight_values(self, mole_fraction):
        """
        Returns
        -------
        ParameterGradient
            The weighted value.
        """
        return ParameterGradient(self._value.key,
                                 self._value.value * mole_fraction)