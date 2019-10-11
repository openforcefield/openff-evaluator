"""
A collection of protocols for reweighting cached simulation data.
"""
import copy
import logging
import re
from os import path

import numpy as np
import typing
from simtk import openmm
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.forcefield import ForceFieldSource, SmirnoffForceFieldSource
from propertyestimator.properties.properties import ParameterGradientKey, ParameterGradient
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import pint_quantity_to_openmm, setup_platform_with_resources, \
    openmm_quantity_to_pint, disable_pbc
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.workflow.decorators import protocol_input, protocol_output, UNDEFINED
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class GradientReducedPotentials(BaseProtocol):
    """A protocol to estimates the the reduced potential of the configurations
    of a trajectory using reverse and forward perturbed simulation parameters for
    use with estimating reweighted gradients using the central difference method.
    """

    reference_force_field_paths = protocol_input(
        docstring='A list of paths to the force field files which were '
                  'originally used to generate the configurations.',
        type_hint=list,
        default_value=UNDEFINED
    )
    force_field_path = protocol_input(
        docstring='The path to the force field which contains the parameters to '
                  'differentiate the observable with respect to.',
        type_hint=str,
        default_value=UNDEFINED
    )

    reference_statistics_path = protocol_input(
        docstring='An optional path to the statistics array which was '
                  'generated alongside the observable of interest, which will '
                  'be used to correct the potential energies at the reverse '
                  'and forward states. This is only really needed when the '
                  'observable of interest is an energy.',
        type_hint=str,
        default_value=UNDEFINED,
        optional=True
    )

    enable_pbc = protocol_input(
        docstring='If true, periodic boundary conditions will be enabled when '
                  're-evaluating the reduced potentials.',
        type_hint=bool,
        default_value=True
    )

    substance = protocol_input(
        docstring='The substance which describes the composition of the system.',
        type_hint=Substance,
        default_value=UNDEFINED
    )
    thermodynamic_state = protocol_input(
        docstring='The thermodynamic state to estimate the gradients at.',
        type_hint=ThermodynamicState,
        default_value=UNDEFINED
    )

    coordinate_file_path = protocol_input(
        docstring='A path to a PDB coordinate file which describes the topology of '
                  'the system.',
        type_hint=str,
        default_value=UNDEFINED
    )
    trajectory_file_path = protocol_input(
        docstring='A path to the trajectory of configurations',
        type_hint=str,
        default_value=UNDEFINED
    )

    parameter_key = protocol_input(
        docstring='The key of the parameter to differentiate with respect to.',
        type_hint=ParameterGradientKey,
        default_value=UNDEFINED
    )

    perturbation_scale = protocol_input(
        docstring='The amount to perturb the parameter by, such that '
                  'p_new = p_old * (1 +/- `perturbation_scale`)',
        type_hint=float,
        default_value=1.0e-4
    )

    use_subset_of_force_field = protocol_input(
        docstring='If true, the reduced potential will be estimated using '
                  'an OpenMM system which only contains the parameter of '
                  'interest',
        type_hint=bool,
        default_value=True
    )

    effective_sample_indices = protocol_input(
        docstring='This a placeholder input which is not currently implemented.',
        type_hint=list,
        default_value=UNDEFINED,
        optional=True
    )

    reference_potential_paths = protocol_output(
        docstring='File paths to the reduced potentials evaluated using each '
                  'of the reference force fields.',
        type_hint=list
    )
    reverse_potentials_path = protocol_output(
        docstring='A file path to the energies evaluated using the parameters'
                  'perturbed in the reverse direction.',
        type_hint=str
    )
    forward_potentials_path = protocol_output(
        docstring='A file path to the energies evaluated using the parameters'
                  'perturbed in the forward direction.',
        type_hint=str
    )

    reverse_parameter_value = protocol_output(
        docstring='The value of the parameter perturbed in the reverse '
                  'direction.',
        type_hint=unit.Quantity
    )
    forward_parameter_value = protocol_output(
        docstring='The value of the parameter perturbed in the forward '
                  'direction.',
        type_hint=unit.Quantity
    )

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

        parameter_tag = self.parameter_key.tag
        parameter_smirks = self.parameter_key.smirks
        parameter_attribute = self.parameter_key.attribute

        original_handler = original_force_field.get_parameter_handler(parameter_tag)
        original_parameter = original_handler.parameters[parameter_smirks]

        if self.use_subset_of_force_field:

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

        if not self.enable_pbc:
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

        temperature = pint_quantity_to_openmm(self.thermodynamic_state.temperature)
        beta = 1.0 / (simtk_unit.BOLTZMANN_CONSTANT_kB * temperature)

        pressure = pint_quantity_to_openmm(self.thermodynamic_state.pressure)

        for frame_index in range(trajectory.n_frames):

            positions = trajectory.xyz[frame_index]
            box_vectors = trajectory.openmm_boxes(frame_index)

            if self.enable_pbc:
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

        logging.info(f'Calculating the reduced gradient potentials for {self.parameter_key}: {self._id}')

        if len(self.reference_force_field_paths) != 1 and self.use_subset_of_force_field:

            return PropertyEstimatorException(directory, 'A single reference force field must be '
                                                         'provided when calculating the reduced '
                                                         'potentials using a subset of the full force')

        if len(self.reference_statistics_path) <= 0 and self.use_subset_of_force_field:

            return PropertyEstimatorException(directory, 'The path to the statistics evaluated using '
                                                         'the full force field must be provided.')

        with open(self.force_field_path) as file:
            target_force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(target_force_field_source, SmirnoffForceFieldSource):

            return PropertyEstimatorException(directory, 'Only SMIRNOFF force fields are supported by '
                                                         'this protocol.')

        target_force_field = target_force_field_source.to_force_field()

        trajectory = mdtraj.load_dcd(self.trajectory_file_path,
                                     self.coordinate_file_path)

        unique_molecules = []

        for component in self.substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)
            unique_molecules.append(molecule)

        pdb_file = app.PDBFile(self.coordinate_file_path)
        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)

        # If we are using only a subset of the system object, load in the reference
        # statistics containing the full system energies to correct the output
        # forward and reverse potential energies.
        reference_statistics = None
        subset_energy_corrections = None

        if self.use_subset_of_force_field:
            reference_statistics = StatisticsArray.from_pandas_csv(self.reference_statistics_path)

        # Compute the reduced reference energy if any reference force field files
        # have been provided.
        for index, reference_force_field_path in enumerate(self.reference_force_field_paths):

            with open(reference_force_field_path) as file:
                reference_force_field_source = ForceFieldSource.parse_json(file.read())

            if not isinstance(reference_force_field_source, SmirnoffForceFieldSource):
                return PropertyEstimatorException(directory, 'Only SMIRNOFF force fields are supported by '
                                                             'this protocol.')

            reference_force_field = reference_force_field_source.to_force_field()
            reference_system, _ = self._build_reduced_system(reference_force_field, topology)

            reference_potentials_path = path.join(directory, f'reference_{index}.csv')

            self._evaluate_reduced_potential(reference_system, trajectory,
                                             reference_potentials_path,
                                             available_resources)

            self.reference_potential_paths.append(reference_potentials_path)

            if reference_statistics is not None:

                subset_energies = StatisticsArray.from_pandas_csv(reference_potentials_path)
                subset_energy_corrections = (reference_statistics[ObservableType.PotentialEnergy] -
                                             subset_energies[ObservableType.PotentialEnergy])

                subset_energies[ObservableType.PotentialEnergy] = reference_statistics[ObservableType.PotentialEnergy]
                subset_energies.to_pandas_csv(reference_potentials_path)

        # Build the slightly perturbed system.
        reverse_system, self.reverse_parameter_value = self._build_reduced_system(target_force_field,
                                                                                  topology,
                                                                                   -self.perturbation_scale)

        forward_system, self.forward_parameter_value = self._build_reduced_system(target_force_field,
                                                                                  topology,
                                                                                  self.perturbation_scale)

        self.reverse_parameter_value = openmm_quantity_to_pint(self.reverse_parameter_value)
        self.forward_parameter_value = openmm_quantity_to_pint(self.forward_parameter_value)

        # Calculate the reduced potentials.
        self.reverse_potentials_path = path.join(directory, 'reverse.csv')
        self.forward_potentials_path = path.join(directory, 'forward.csv')

        self._evaluate_reduced_potential(reverse_system, trajectory, self.reverse_potentials_path,
                                         available_resources, subset_energy_corrections)
        self._evaluate_reduced_potential(forward_system, trajectory, self.forward_potentials_path,
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

    parameter_key = protocol_input(
        docstring='The key of the parameter to differentiate with respect to.',
        type_hint=ParameterGradientKey,
        default_value=UNDEFINED
    )

    reverse_observable_value = protocol_input(
        docstring='The value of the observable evaluated using the parameters'
                  'perturbed in the reverse direction.',
        type_hint=typing.Union[unit.Quantity, EstimatedQuantity],
        default_value=UNDEFINED
    )
    forward_observable_value = protocol_input(
        docstring='The value of the observable evaluated using the parameters'
                  'perturbed in the forward direction.',
        type_hint=typing.Union[unit.Quantity, EstimatedQuantity],
        default_value=UNDEFINED
    )

    reverse_parameter_value = protocol_input(
        docstring='The value of the parameter perturbed in the reverse '
                  'direction.',
        type_hint=unit.Quantity,
        default_value=UNDEFINED
    )
    forward_parameter_value = protocol_input(
        docstring='The value of the parameter perturbed in the forward '
                  'direction.',
        type_hint=unit.Quantity,
        default_value=UNDEFINED
    )

    gradient = protocol_output(
        docstring='The estimated gradient',
        type_hint=ParameterGradient
    )

    def execute(self, directory, available_resources):

        if self.forward_parameter_value < self.reverse_parameter_value:

            return PropertyEstimatorException(f'The forward parameter value ({self.forward_parameter_value}) must '
                                              f'be larger than the reverse value ({self.reverse_parameter_value}).')

        reverse_value = self.reverse_observable_value
        forward_value = self.forward_observable_value

        if isinstance(reverse_value, EstimatedQuantity):
            reverse_value = reverse_value.value

        if isinstance(forward_value, EstimatedQuantity):
            forward_value = forward_value.value

        gradient = ((forward_value - reverse_value) /
                    (self.forward_parameter_value - self.reverse_parameter_value))

        self.gradient = ParameterGradient(self.parameter_key, gradient)

        return self._get_output_dictionary()
