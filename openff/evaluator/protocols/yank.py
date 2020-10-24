"""
A collection of protocols for performing free energy calculations
using the YANK package.
"""
import abc
import logging
import os
import shutil
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pint
import yaml

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import (
    ParameterGradient,
    ParameterGradientKey,
    SmirnoffForceFieldSource,
)
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.openmm import _compute_gradients
from openff.evaluator.substances import Component, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import timeseries
from openff.evaluator.utils.observables import (
    Observable,
    ObservableArray,
    ObservableFrame,
    ObservableType,
)
from openff.evaluator.utils.openmm import (
    disable_pbc,
    openmm_quantity_to_pint,
    pint_quantity_to_openmm,
    setup_platform_with_resources,
)
from openff.evaluator.utils.timeseries import (
    TimeSeriesStatistics,
    get_uncorrelated_indices,
)
from openff.evaluator.utils.utils import temporarily_change_directory
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)

if TYPE_CHECKING:
    import mdtraj
    from openforcefield.topology import Topology
    from openforcefield.typing.engines.smirnoff.forcefield import ForceField

logger = logging.getLogger(__name__)


class BaseYankProtocol(Protocol, abc.ABC):
    """An abstract base class for protocols which will performs a set of
    alchemical free energy simulations using the YANK framework.
    """

    thermodynamic_state = InputAttribute(
        docstring="The state at which to run the calculations.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    number_of_equilibration_iterations = InputAttribute(
        docstring="The number of iterations used for equilibration before production "
        "run. Only post-equilibration iterations are written to file.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1,
    )
    number_of_iterations = InputAttribute(
        docstring="The number of YANK iterations to perform.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=5000,
    )
    steps_per_iteration = InputAttribute(
        docstring="The number of steps per YANK iteration to perform.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=500,
    )
    checkpoint_interval = InputAttribute(
        docstring="The number of iterations between saving YANK checkpoint files.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1,
    )

    timestep = InputAttribute(
        docstring="The length of the timestep to take.",
        type_hint=pint.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2 * unit.femtosecond,
    )

    verbose = InputAttribute(
        docstring="Controls whether or not to run YANK at high verbosity.",
        type_hint=bool,
        default_value=False,
    )
    setup_only = InputAttribute(
        docstring="If true, YANK will only create and validate the setup files, "
        "but not actually run any simulations. This argument is mainly "
        "only to be used for testing purposes.",
        type_hint=bool,
        default_value=False,
    )

    gradient_parameters = InputAttribute(
        docstring="An optional list of parameters to differentiate the estimated "
        "free energy with respect to.",
        type_hint=list,
        default_value=lambda: list(),
    )

    free_energy_difference = OutputAttribute(
        docstring="The estimated free energy difference between the two phases of"
        "interest.",
        type_hint=Observable,
    )

    def __init__(self, protocol_id):
        super(BaseYankProtocol, self).__init__(protocol_id)
        self._analysed_output = None

    @staticmethod
    def _get_residue_names_from_role(substances, coordinate_path, role):
        """Returns a list of all of the residue names of
        components which have been assigned a given role.

        Parameters
        ----------
        substances: list of Substance
            The substances which contains the components.
        coordinate_path: str
            The path to the coordinates which describe the systems
            topology.
        role: Component.Role, optional
            The role of the component to identify.

        Returns
        -------
        set of str
            The identified residue names.
        """

        from openforcefield.topology import Molecule, Topology
        from simtk.openmm import app

        if role is None:
            return "all"

        unique_molecules = [
            Molecule.from_smiles(component.smiles)
            for substance in substances
            for component in substance.components
        ]

        openmm_topology = app.PDBFile(coordinate_path).topology
        topology = Topology.from_openmm(openmm_topology, unique_molecules)

        # Determine the smiles of all molecules in the system. We need to use
        # the toolkit to re-generate the smiles as later we will compare these
        # against more toolkit generated smiles.
        components = [
            component
            for substance in substances
            for component in substance.components
            if component.role == role
        ]

        component_smiles = [
            Molecule.from_smiles(component.smiles).to_smiles()
            for component in components
        ]

        residue_names = set()

        all_openmm_atoms = list(openmm_topology.atoms())

        # Find the resiude names of the molecules which have the correct
        # role.
        for topology_molecule in topology.topology_molecules:

            molecule_smiles = topology_molecule.reference_molecule.to_smiles()

            if molecule_smiles not in component_smiles:
                continue

            molecule_residue_names = set(
                [
                    all_openmm_atoms[topology_atom.topology_atom_index].residue.name
                    for topology_atom in topology_molecule.atoms
                ]
            )

            assert len(molecule_residue_names) == 1
            residue_names.update(molecule_residue_names)

        return residue_names

    @staticmethod
    def _get_dsl_from_role(substances, coordinate_path, role):
        """Returns an MDTraj DSL string which identifies those
        atoms which belong to components flagged with a specific
        role.

        Parameters
        ----------
        substances: list of Substance
            The substances which contains the components.
        coordinate_path: str
            The path to the coordinates which describe the systems
            topology.
        role: Component.Role, optional
            The role of the component to identify.

        Returns
        -------
        str
            The DSL string.
        """

        residue_names = BaseYankProtocol._get_residue_names_from_role(
            substances, coordinate_path, role
        )

        dsl_string = " or ".join(
            [f"resname {residue_name}" for residue_name in residue_names]
        )
        return dsl_string

    def _get_options_dictionary(self, available_resources):
        """Returns a dictionary of options which will be serialized
        to a yaml file and passed to YANK.

        Parameters
        ----------
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK options.
        """

        from openforcefield.utils import quantity_to_string

        platform_name = "CPU"

        if available_resources.number_of_gpus > 0:

            # A platform which runs on GPUs has been requested.
            from openff.evaluator.backends import ComputeResources

            toolkit_enum = ComputeResources.GPUToolkit(
                available_resources.preferred_gpu_toolkit
            )

            # A platform which runs on GPUs has been requested.
            platform_name = (
                "CUDA"
                if toolkit_enum == ComputeResources.GPUToolkit.CUDA
                else ComputeResources.GPUToolkit.OpenCL
            )

        return {
            "verbose": self.verbose,
            "output_dir": ".",
            "temperature": quantity_to_string(
                pint_quantity_to_openmm(self.thermodynamic_state.temperature)
            ),
            "pressure": quantity_to_string(
                pint_quantity_to_openmm(self.thermodynamic_state.pressure)
            ),
            "minimize": True,
            "number_of_equilibration_iterations": (
                self.number_of_equilibration_iterations
            ),
            "default_number_of_iterations": self.number_of_iterations,
            "default_nsteps_per_iteration": self.steps_per_iteration,
            "checkpoint_interval": self.checkpoint_interval,
            "default_timestep": quantity_to_string(
                pint_quantity_to_openmm(self.timestep)
            ),
            "annihilate_electrostatics": True,
            "annihilate_sterics": False,
            "platform": platform_name,
        }

    @abc.abstractmethod
    def _get_system_dictionary(self):
        """Returns a dictionary of the system which will be serialized
        to a yaml file and passed to YANK. Only a single system may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK systems.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_protocol_dictionary(self):
        """Returns a dictionary of the protocol which will be serialized
        to a yaml file and passed to YANK. Only a single protocol may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK protocol.
        """
        raise NotImplementedError()

    def _get_experiments_dictionary(self):
        """Returns a dictionary of the experiments which will be serialized
        to a yaml file and passed to YANK. Only a single experiment may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK experiment.
        """

        system_dictionary = self._get_system_dictionary()
        system_key = next(iter(system_dictionary))

        protocol_dictionary = self._get_protocol_dictionary()
        protocol_key = next(iter(protocol_dictionary))

        return {"system": system_key, "protocol": protocol_key}

    def _get_full_input_dictionary(self, available_resources):
        """Returns a dictionary of the full YANK inputs which will be serialized
        to a yaml file and passed to YANK

        Parameters
        ----------
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK input file.
        """

        return {
            "options": self._get_options_dictionary(available_resources),
            "systems": self._get_system_dictionary(),
            "protocols": self._get_protocol_dictionary(),
            "experiments": self._get_experiments_dictionary(),
        }

    def _time_series_statistics(self, phase: str) -> TimeSeriesStatistics:
        """Returns the time series statistics (such as the equilibration time) for
        a particular phase."""

        equilibration = self._analysed_output["equilibration"][phase]
        n_expected = self.number_of_iterations

        uncorrelated_indices = get_uncorrelated_indices(
            n_expected - equilibration["discarded_from_start"],
            equilibration["subsample_rate"],
        )

        time_series_statistics = TimeSeriesStatistics(
            n_total_points=n_expected,
            n_uncorrelated_points=len(uncorrelated_indices),
            statistical_inefficiency=equilibration["subsample_rate"],
            equilibration_index=equilibration["discarded_from_start"],
        )

        return time_series_statistics

    @staticmethod
    def _extract_trajectory(
        checkpoint_path: str,
        output_trajectory_path: Optional[str],
        statistics: TimeSeriesStatistics,
        state_index: int = 0,
    ) -> "mdtraj.Trajectory":
        """Extracts the stored trajectory of the 'initial' state from a
        yank `.nc` checkpoint file and stores it to disk as a `.dcd` file.

        Parameters
        ----------
        checkpoint_path
            The path to the yank `.nc` file
        output_trajectory_path
            The path to optionally store the extracted trajectory at.
        statistics
            Statistics about the time series to use to decorrelate and remove
            un-equilibrated samples.
        """

        from yank.analyze import extract_trajectory

        trajectory = extract_trajectory(
            checkpoint_path, state_index=state_index, image_molecules=True
        )
        trajectory = trajectory[statistics.equilibration_index :]

        uncorrelated_indices = timeseries.get_uncorrelated_indices(
            statistics.n_total_points - statistics.equilibration_index,
            statistics.statistical_inefficiency,
        )

        trajectory = trajectory[np.array(uncorrelated_indices)]

        if output_trajectory_path is not None:
            trajectory.save_dcd(output_trajectory_path)

        return trajectory

    @staticmethod
    def _run_yank(directory, available_resources, setup_only) -> Dict[str, Any]:
        """Runs YANK within the specified directory which contains a `yank.yaml`
        input file.

        Parameters
        ----------
        directory: str
            The directory within which to run yank.
        available_resources: ComputeResources
            The compute resources available to yank.
        setup_only: bool
            If true, YANK will only create and validate the setup files,
            but not actually run any simulations. This argument is mainly
            only to be used for testing purposes.

        Returns
        -------
            The analysed output of the yank calculation.
        """

        from yank.analyze import ExperimentAnalyzer
        from yank.experiment import ExperimentBuilder

        with temporarily_change_directory(directory):

            # Set the default properties on the desired platform
            # before calling into yank.
            setup_platform_with_resources(available_resources)

            exp_builder = ExperimentBuilder("yank.yaml")

            if setup_only is True:

                from simtk import unit as simtk_unit

                return {
                    "free_energy": {
                        "free_energy_diff_unit": 0.0 * simtk_unit.kilojoules_per_mole,
                        "free_energy_diff_error_unit": 0.0
                        * simtk_unit.kilojoules_per_mole,
                    }
                }

            exp_builder.run_experiments()

            analyzer = ExperimentAnalyzer("experiments")
            analysed_output = analyzer.auto_analyze()

        return analysed_output

    @staticmethod
    def _run_yank_as_process(queue, directory, available_resources, setup_only):
        """A wrapper around the `_run_yank` method which takes
        a `multiprocessing.Queue` as input, thereby allowing it
        to be launched from a separate process and still return
        it's output back to the main process.

        Parameters
        ----------
        queue: multiprocessing.Queue
            The queue object which will communicate with the
            launched process.
        directory: str
            The directory within which to run yank.
        available_resources: ComputeResources
            The compute resources available to yank.
        setup_only: bool
            If true, YANK will only create and validate the setup files,
            but not actually run any simulations. This argument is mainly
            only to be used for testing purposes.
        """

        analysed_output = None
        exception = None

        try:
            analysed_output = BaseYankProtocol._run_yank(
                directory, available_resources, setup_only
            )
        except Exception as e:
            exception = e

        queue.put((analysed_output, exception))

    def _compute_state_energy_gradients(
        self,
        trajectory: "mdtraj.Trajectory",
        topology: "Topology",
        force_field: "ForceField",
        enable_pbc: bool,
        compute_resources: ComputeResources,
    ) -> List[ParameterGradient]:
        """Computes the value of <dU / d theta> for a specified trajectory and for
        each force field parameter (theta) of interest.

        Parameters
        ----------
        trajectory
            The trajectory of interest.
        topology
            The topology of the system.
        force_field
            The force field containing the parameters of interest.
        enable_pbc
            Whether periodic boundary conditions should be enabled when evaluating
            the potential energies of each frame and their gradients.
        compute_resources
            The resources available when computing the gradients.

        Returns
        -------
            The average gradient of the potential energy with respect to each force
            field parameter of interest.
        """

        # Mock an observable frame to store the gradients in
        observables = ObservableFrame(
            {
                ObservableType.PotentialEnergy: ObservableArray(
                    value=np.zeros((len(trajectory), 1)) * unit.kilojoule / unit.mole
                )
            }
        )

        # Compute the gradient in the first solvent.
        _compute_gradients(
            self.gradient_parameters,
            observables,
            force_field,
            self.thermodynamic_state,
            topology,
            trajectory,
            compute_resources,
            enable_pbc,
        )

        return [
            ParameterGradient(key=gradient.key, value=gradient.value.mean().item())
            for gradient in observables[ObservableType.PotentialEnergy].gradients
        ]

    def _execute(self, directory, available_resources):

        yaml_filename = os.path.join(directory, "yank.yaml")

        # Create the yank yaml input file from a dictionary of options.
        with open(yaml_filename, "w") as file:
            yaml.dump(
                self._get_full_input_dictionary(available_resources),
                file,
                sort_keys=False,
            )

        setup_only = self.setup_only

        # Yank is not safe to be called from anything other than the main thread.
        # If the current thread is not detected as the main one, then yank should
        # be spun up in a new process which should itself be safe to run yank in.
        if threading.current_thread() is threading.main_thread():
            logger.info("Launching YANK in the main thread.")
            analysed_output = self._run_yank(directory, available_resources, setup_only)
        else:

            from multiprocessing import Process, Queue

            logger.info("Launching YANK in a new process.")

            # Create a queue to pass the results back to the main process.
            queue = Queue()
            # Create the process within which yank will run.
            # noinspection PyTypeChecker
            process = Process(
                target=BaseYankProtocol._run_yank_as_process,
                args=[queue, directory, available_resources, setup_only],
            )

            # Start the process and gather back the output.
            process.start()
            analysed_output, exception = queue.get()
            process.join()

            if exception is not None:
                raise exception

        free_energy_difference = analysed_output["free_energy"]["free_energy_diff_unit"]
        free_energy_difference_std = analysed_output["free_energy"][
            "free_energy_diff_error_unit"
        ]

        self._analysed_output = analysed_output

        self.free_energy_difference = Observable(
            value=openmm_quantity_to_pint(free_energy_difference).plus_minus(
                openmm_quantity_to_pint(free_energy_difference_std)
            )
        )

    def validate(self, attribute_type=None):
        super(BaseYankProtocol, self).validate(attribute_type)

        if self.checkpoint_interval != 1:
            raise ValueError(
                "The checkpoint interval must currently be set to one due to a bug in "
                "how YANK extracts trajectories from checkpoint files."
            )


@workflow_protocol()
class LigandReceptorYankProtocol(BaseYankProtocol):
    """A protocol for performing ligand-receptor alchemical free energy
    calculations using the YANK framework.
    """

    class RestraintType(Enum):
        """The types of ligand restraints available within yank."""

        Harmonic = "Harmonic"
        FlatBottom = "FlatBottom"

    ligand_residue_name = InputAttribute(
        docstring="The residue name of the ligand.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    receptor_residue_name = InputAttribute(
        docstring="The residue name of the receptor.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    solvated_ligand_coordinates = InputAttribute(
        docstring="The file path to the solvated ligand coordinates.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    solvated_ligand_system = InputAttribute(
        docstring="The parameterized solvated ligand system object.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    solvated_complex_coordinates = InputAttribute(
        docstring="The file path to the solvated complex coordinates.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    solvated_complex_system = InputAttribute(
        docstring="The parameterized solvated complex system object.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    force_field_path = InputAttribute(
        docstring="The path to the force field which defines the charge method "
        "to use for the calculation.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    apply_restraints = InputAttribute(
        docstring="Determines whether the ligand should be explicitly restrained to the "
        "receptor in order to stop the ligand from temporarily unbinding.",
        type_hint=bool,
        default_value=True,
    )
    restraint_type = InputAttribute(
        docstring="The type of ligand restraint applied, provided that "
        "`apply_restraints` is `True`",
        type_hint=RestraintType,
        default_value=RestraintType.Harmonic,
    )

    ligand_electrostatic_lambdas = InputAttribute(
        docstring="The list of electrostatic alchemical states that YANK should sample "
        "at when calculating the free energy of the solvated ligand. If no option is "
        "set, YANK will use `trailblaze` algorithm to determine this option "
        "automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    ligand_steric_lambdas = InputAttribute(
        docstring="The list of steric alchemical states that YANK should sample "
        "at when calculating the free energy of the solvated ligand. If no option is "
        "set, YANK will use `trailblaze` algorithm to determine this option "
        "automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    complex_electrostatic_lambdas = InputAttribute(
        docstring="The list of electrostatic alchemical states that YANK should sample "
        "at when calculating the free energy of the ligand in complex with the "
        "receptor. If no option is set, YANK will use `trailblaze` algorithm to "
        "determine this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    complex_steric_lambdas = InputAttribute(
        docstring="The list of steric alchemical states that YANK should sample "
        "at when calculating the free energy of the ligand in complex with the "
        "receptor. If no option is set, YANK will use `trailblaze` algorithm to "
        "determine this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )

    solvated_ligand_trajectory_path = OutputAttribute(
        docstring="The file path to the generated ligand trajectory.", type_hint=str
    )
    solvated_complex_trajectory_path = OutputAttribute(
        docstring="The file path to the generated ligand trajectory.", type_hint=str
    )

    def __init__(self, protocol_id):
        """Constructs a new LigandReceptorYankProtocol object."""

        super().__init__(protocol_id)

        self._local_ligand_coordinates = "ligand.pdb"
        self._local_ligand_system = "ligand.xml"

        self._local_complex_coordinates = "complex.pdb"
        self._local_complex_system = "complex.xml"

    def _get_solvent_dictionary(self):
        """Returns a dictionary of the solvent which will be serialized
        to a yaml file and passed to YANK. In most cases, this should
        just be passing force field settings over, such as PME settings.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK solvents.
        """

        with open(self.force_field_path, "r") as file:
            force_field_source = SmirnoffForceFieldSource.parse_json(file.read())

        force_field = force_field_source.to_force_field()
        charge_method = force_field.get_parameter_handler("Electrostatics").method

        if charge_method.lower() != "pme":
            raise ValueError("Currently only PME electrostatics are supported.")

        return {"default": {"nonbonded_method": charge_method}}

    def _get_system_dictionary(self):

        solvent_dictionary = self._get_solvent_dictionary()
        solvent_key = next(iter(solvent_dictionary))

        host_guest_dictionary = {
            "phase1_path": [
                self._local_complex_system,
                self._local_complex_coordinates,
            ],
            "phase2_path": [self._local_ligand_system, self._local_ligand_coordinates],
            "ligand_dsl": f"resname {self.ligand_residue_name}",
            "solvent": solvent_key,
        }

        return {"host-guest": host_guest_dictionary}

    def _get_protocol_dictionary(self):

        ligand_protocol_dictionary = {
            "lambda_electrostatics": self.ligand_electrostatic_lambdas,
            "lambda_sterics": self.ligand_steric_lambdas,
        }

        if (
            self.ligand_electrostatic_lambdas == UNDEFINED
            and self.ligand_steric_lambdas == UNDEFINED
        ):

            ligand_protocol_dictionary = "auto"

        elif (
            self.ligand_electrostatic_lambdas != UNDEFINED
            and self.ligand_steric_lambdas == UNDEFINED
        ) or (
            self.ligand_electrostatic_lambdas == UNDEFINED
            and self.ligand_steric_lambdas != UNDEFINED
        ):

            raise ValueError(
                "Either both of `ligand_electrostatic_lambdas` and "
                "`ligand_steric_lambdas` must be set, or neither "
                "must be set."
            )

        complex_protocol_dictionary = {
            "lambda_electrostatics": self.complex_electrostatic_lambdas,
            "lambda_sterics": self.complex_steric_lambdas,
        }

        if (
            self.complex_electrostatic_lambdas == UNDEFINED
            and self.complex_steric_lambdas == UNDEFINED
        ):

            complex_protocol_dictionary = "auto"

        elif (
            self.complex_electrostatic_lambdas != UNDEFINED
            and self.complex_steric_lambdas == UNDEFINED
        ) or (
            self.complex_electrostatic_lambdas == UNDEFINED
            and self.complex_steric_lambdas != UNDEFINED
        ):

            raise ValueError(
                "Either both of `complex_electrostatic_lambdas` and "
                "`complex_steric_lambdas` must be set, or neither "
                "must be set."
            )

        absolute_binding_dictionary = {
            "complex": {"alchemical_path": complex_protocol_dictionary},
            "solvent": {"alchemical_path": ligand_protocol_dictionary},
        }

        return {"absolute_binding_dictionary": absolute_binding_dictionary}

    def _get_experiments_dictionary(self):

        experiments_dictionary = super(
            LigandReceptorYankProtocol, self
        )._get_experiments_dictionary()

        if self.apply_restraints:

            ligand_dsl = f"(resname {self.ligand_residue_name}) and (mass > 1.5)"
            receptor_dsl = f"(resname {self.receptor_residue_name}) and (mass > 1.5)"

            experiments_dictionary["restraint"] = {
                "restrained_ligand_atoms": ligand_dsl,
                "restrained_receptor_atoms": receptor_dsl,
                "type": self.restraint_type.value,
            }

        return experiments_dictionary

    def _get_full_input_dictionary(self, available_resources):

        full_dictionary = super(
            LigandReceptorYankProtocol, self
        )._get_full_input_dictionary(available_resources)
        full_dictionary["solvents"] = self._get_solvent_dictionary()

        return full_dictionary

    def _execute(self, directory, available_resources):

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(
            self.solvated_ligand_coordinates,
            os.path.join(directory, self._local_ligand_coordinates),
        )
        shutil.copyfile(
            self.solvated_ligand_system.system_path,
            os.path.join(directory, self._local_ligand_system),
        )

        shutil.copyfile(
            self.solvated_complex_coordinates,
            os.path.join(directory, self._local_complex_coordinates),
        )
        shutil.copyfile(
            self.solvated_complex_system.system_path,
            os.path.join(directory, self._local_complex_system),
        )

        super(LigandReceptorYankProtocol, self)._execute(directory, available_resources)

        if self.setup_only:
            return

        ligand_yank_path = os.path.join(directory, "experiments", "solvent.nc")
        complex_yank_path = os.path.join(directory, "experiments", "complex.nc")

        self.solvated_ligand_trajectory_path = os.path.join(directory, "ligand.dcd")
        self.solvated_complex_trajectory_path = os.path.join(directory, "complex.dcd")

        self._extract_trajectory(
            ligand_yank_path,
            self.solvated_ligand_trajectory_path,
            self._time_series_statistics("solvent"),
        )
        self._extract_trajectory(
            complex_yank_path,
            self.solvated_complex_trajectory_path,
            self._time_series_statistics("complex"),
        )


@workflow_protocol()
class SolvationYankProtocol(BaseYankProtocol):
    """A protocol for estimating the change in free energy upon transferring a solute
    into a solvent (referred to as solvent 1) from a second solvent (referred to as
    solvent 2) by performing an alchemical free energy calculation using the YANK
    framework.

    This protocol can be used for box solvation free energies (setting the `solvent_1`
    input to the solvent of interest and setting `solvent_2` as an empty `Substance`) or
    transfer free energies (setting both the `solvent_1` and `solvent_2` inputs to
    different solvents).
    """

    solute = InputAttribute(
        docstring="The substance describing the composition of "
        "the solute. This should include the solute "
        "molecule as well as any counter ions.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    solvent_1 = InputAttribute(
        docstring="The substance describing the composition of the first solvent.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    solvent_2 = InputAttribute(
        docstring="The substance describing the composition of the second solvent.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    solution_1_coordinates = InputAttribute(
        docstring="The file path to the coordinates of the solute embedded in the "
        "first solvent.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    solution_1_system = InputAttribute(
        docstring="The parameterized system object of the solute embedded in the "
        "first solvent.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    solution_2_coordinates = InputAttribute(
        docstring="The file path to the coordinates of the solute embedded in the "
        "second solvent.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    solution_2_system = InputAttribute(
        docstring="The parameterized system object of the solute embedded in the "
        "second solvent.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    electrostatic_lambdas_1 = InputAttribute(
        docstring="The list of electrostatic alchemical states that YANK should sample "
        "at. These values will be passed to the YANK `lambda_electrostatics` option. "
        "If no option is set, YANK will use `trailblaze` algorithm to determine "
        "this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    steric_lambdas_1 = InputAttribute(
        docstring="The list of steric alchemical states that YANK should sample at. "
        "These values will be passed to the YANK `lambda_sterics` option. "
        "If no option is set, YANK will use `trailblaze` algorithm to determine "
        "this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    electrostatic_lambdas_2 = InputAttribute(
        docstring="The list of electrostatic alchemical states that YANK should sample "
        "at. These values will be passed to the YANK `lambda_electrostatics` option. "
        "If no option is set, YANK will use `trailblaze` algorithm to determine "
        "this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )
    steric_lambdas_2 = InputAttribute(
        docstring="The list of steric alchemical states that YANK should sample at. "
        "These values will be passed to the YANK `lambda_sterics` option. "
        "If no option is set, YANK will use `trailblaze` algorithm to determine "
        "this option automatically.",
        type_hint=list,
        optional=True,
        default_value=UNDEFINED,
    )

    solution_1_free_energy = OutputAttribute(
        docstring="The free energy change of transforming the an ideal solute molecule "
        "into a fully interacting molecule in the first solvent.",
        type_hint=Observable,
    )
    solvent_1_coordinate_path = OutputAttribute(
        docstring="The file path to the coordinates of only the first solvent.",
        type_hint=str,
    )
    solvent_1_trajectory_path = OutputAttribute(
        docstring="The file path to the trajectory containing only the first solvent.",
        type_hint=str,
    )
    solution_1_trajectory_path = OutputAttribute(
        docstring="The file path to the trajectory containing the solute in the first "
        "solvent.",
        type_hint=str,
    )

    solution_2_free_energy = OutputAttribute(
        docstring="The free energy change of transforming the an ideal solute molecule "
        "into a fully interacting molecule in the second solvent.",
        type_hint=Observable,
    )
    solvent_2_coordinate_path = OutputAttribute(
        docstring="The file path to the coordinates of only the second solvent.",
        type_hint=str,
    )
    solvent_2_trajectory_path = OutputAttribute(
        docstring="The file path to the trajectory containing only the second solvent.",
        type_hint=str,
    )
    solution_2_trajectory_path = OutputAttribute(
        docstring="The file path to the trajectory containing the solute in the second "
        "solvent.",
        type_hint=str,
    )

    free_energy_difference = OutputAttribute(
        docstring="The estimated free energy difference between the solute in the"
        "first solvent and the second solvent (i.e. ΔG = ΔG_1 - ΔG_2).",
        type_hint=Observable,
    )

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._local_solution_1_coordinates = "solvent_1.pdb"
        self._local_solution_1_system = "solvent_1.xml"

        self._local_solution_2_coordinates = "solvent_2.pdb"
        self._local_solution_2_system = "solvent_2.xml"

    def _get_system_dictionary(self):

        solvent_1_dsl = self._get_dsl_from_role(
            [self.solute, self.solvent_1],
            self.solution_1_coordinates,
            Component.Role.Solvent,
        )

        solvent_2_dsl = self._get_dsl_from_role(
            [self.solute, self.solvent_2],
            self.solution_2_coordinates,
            Component.Role.Solvent,
        )

        full_solvent_dsl_components = []

        if len(solvent_1_dsl) > 0:
            full_solvent_dsl_components.append(solvent_1_dsl)
        if len(solvent_2_dsl) > 0:
            full_solvent_dsl_components.append(solvent_2_dsl)

        solvation_system_dictionary = {
            "phase1_path": [
                self._local_solution_1_system,
                self._local_solution_1_coordinates,
            ],
            "phase2_path": [
                self._local_solution_2_system,
                self._local_solution_2_coordinates,
            ],
            "solvent_dsl": " or ".join(full_solvent_dsl_components),
        }

        return {"solvation-system": solvation_system_dictionary}

    def _get_protocol_dictionary(self):

        solvent_1_protocol_dictionary = {
            "lambda_electrostatics": self.electrostatic_lambdas_1,
            "lambda_sterics": self.steric_lambdas_1,
        }

        if (
            self.electrostatic_lambdas_1 == UNDEFINED
            and self.steric_lambdas_1 == UNDEFINED
        ):

            solvent_1_protocol_dictionary = "auto"

        elif (
            self.electrostatic_lambdas_1 != UNDEFINED
            and self.steric_lambdas_1 == UNDEFINED
        ) or (
            self.electrostatic_lambdas_1 == UNDEFINED
            and self.steric_lambdas_1 != UNDEFINED
        ):

            raise ValueError(
                "Either both of `electrostatic_lambdas_1` and "
                "`steric_lambdas_1` must be set, or neither "
                "must be set."
            )

        solvent_2_protocol_dictionary = {
            "lambda_electrostatics": self.electrostatic_lambdas_2,
            "lambda_sterics": self.steric_lambdas_2,
        }

        if (
            self.electrostatic_lambdas_2 == UNDEFINED
            and self.steric_lambdas_2 == UNDEFINED
        ):

            solvent_2_protocol_dictionary = "auto"

        elif (
            self.electrostatic_lambdas_2 != UNDEFINED
            and self.steric_lambdas_2 == UNDEFINED
        ) or (
            self.electrostatic_lambdas_2 == UNDEFINED
            and self.steric_lambdas_2 != UNDEFINED
        ):

            raise ValueError(
                "Either both of `electrostatic_lambdas_2` and "
                "`steric_lambdas_2` must be set, or neither "
                "must be set."
            )

        protocol_dictionary = {
            "solvent1": {"alchemical_path": solvent_1_protocol_dictionary},
            "solvent2": {"alchemical_path": solvent_2_protocol_dictionary},
        }

        return {"solvation-protocol": protocol_dictionary}

    @classmethod
    def _extract_solvent_trajectory(
        cls,
        checkpoint_path: str,
        output_trajectory_path: Optional[str],
        statistics: TimeSeriesStatistics,
        state_index: int = 0,
    ) -> "mdtraj.Trajectory":
        """Extracts the stored trajectory of the from a yank `.nc` checkpoint file,
        removes the solute, and stores it to disk as a `.dcd` file.

        Parameters
        ----------
        checkpoint_path
            The path to the yank `.nc` file
        output_trajectory_path
            The path to optionally store the extracted trajectory at.
        statistics
            Statistics about the time series to use to decorrelate and remove
            un-equilibrated samples.
        state_index
            The state index to extract.
        """

        import openmmtools

        trajectory = cls._extract_trajectory(
            checkpoint_path, None, statistics, state_index
        )

        reporter = None

        try:
            reporter = openmmtools.multistate.MultiStateReporter(
                checkpoint_path, open_mode="r"
            )
            solute_indices = reporter.analysis_particle_indices
        finally:
            if reporter is not None:
                reporter.close()

        solvent_indices = {*range(trajectory.n_atoms)} - set(solute_indices)
        solvent_trajectory = trajectory.atom_slice([*solvent_indices])

        if output_trajectory_path is not None:
            solvent_trajectory.save_dcd(output_trajectory_path)

        return solvent_trajectory

    def _analyze_phase(
        self,
        checkpoint_path: str,
        parameterized_system: ParameterizedSystem,
        phase_name: str,
        available_resources: ComputeResources,
    ) -> Tuple[
        Observable,
        "mdtraj.Trajectory",
        "mdtraj.Trajectory",
        Dict[ParameterGradientKey, ParameterGradient],
        Dict[ParameterGradientKey, ParameterGradient],
    ]:
        """Analyzes a particular phase, extracting the relevant free energies
        and computing the required free energies."""

        from openforcefield.topology import Molecule, Topology

        free_energies = self._analysed_output["free_energy"]

        # Extract the free energy change.
        free_energy = -Observable(
            openmm_quantity_to_pint(
                (
                    free_energies[phase_name]["free_energy_diff"]
                    * free_energies[phase_name]["kT"]
                )
            ).plus_minus(
                openmm_quantity_to_pint(
                    free_energies[phase_name]["free_energy_diff_error"]
                    * free_energies[phase_name]["kT"]
                )
            )
        )

        # Extract the statistical inefficiency of the data.
        time_series_statistics = self._time_series_statistics(phase_name)

        # Extract the solution and solvent trajectories.
        solution_system = parameterized_system

        solution_trajectory = self._extract_trajectory(
            checkpoint_path, None, time_series_statistics
        )
        solvent_trajectory = self._extract_solvent_trajectory(
            checkpoint_path,
            None,
            time_series_statistics,
            self._analysed_output["general"][phase_name]["nstates"] - 1,
        )

        solvent_topology_omm = solvent_trajectory.topology.to_openmm()
        solvent_topology = Topology.from_openmm(
            solvent_topology_omm,
            [
                Molecule.from_smiles(component.smiles)
                for component in solution_system.substance
            ],
        )

        # Optionally compute any gradients.
        if len(self.gradient_parameters) == 0:
            return free_energy, solution_trajectory, solvent_trajectory, {}, {}

        force_field_source = solution_system.force_field

        if not isinstance(force_field_source, SmirnoffForceFieldSource):
            raise ValueError(
                "Derivates can only be computed for systems parameterized with "
                "SMIRNOFF force fields."
            )

        force_field = force_field_source.to_force_field()

        solution_gradients = {
            gradient.key: gradient
            for gradient in self._compute_state_energy_gradients(
                solution_trajectory,
                solution_system.topology,
                force_field,
                solvent_topology.n_topology_atoms != 0,
                available_resources,
            )
        }
        solvent_gradients = {
            gradient.key: gradient
            for gradient in self._compute_state_energy_gradients(
                solvent_trajectory,
                solvent_topology,
                force_field,
                solvent_topology.n_topology_atoms != 0,
                available_resources,
            )
        }

        return (
            free_energy,
            solution_trajectory,
            solvent_trajectory,
            solution_gradients,
            solvent_gradients,
        )

    def _execute(self, directory, available_resources):

        from simtk.openmm import XmlSerializer

        solute_components = [
            component
            for component in self.solute.components
            if component.role == Component.Role.Solute
        ]

        solvent_1_components = [
            component
            for component in self.solvent_1.components
            if component.role == Component.Role.Solvent
        ]

        solvent_2_components = [
            component
            for component in self.solvent_2.components
            if component.role == Component.Role.Solvent
        ]

        if len(solute_components) != 1:
            raise ValueError(
                "There must only be a single component marked as a solute."
            )
        if len(solvent_1_components) == 0 and len(solvent_2_components) == 0:
            raise ValueError("At least one of the solvents must not be vacuum.")

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(
            self.solution_1_coordinates,
            os.path.join(directory, self._local_solution_1_coordinates),
        )
        shutil.copyfile(
            self.solution_1_system.system_path,
            os.path.join(directory, self._local_solution_1_system),
        )

        shutil.copyfile(
            self.solution_2_coordinates,
            os.path.join(directory, self._local_solution_2_coordinates),
        )
        shutil.copyfile(
            self.solution_2_system.system_path,
            os.path.join(directory, self._local_solution_2_system),
        )

        # Disable the pbc of the any solvents which should be treated
        # as vacuum.
        vacuum_system_path = None

        if len(solvent_1_components) == 0:
            vacuum_system_path = self._local_solution_1_system
        elif len(solvent_2_components) == 0:
            vacuum_system_path = self._local_solution_2_system

        if vacuum_system_path is not None:

            logger.info(
                f"Disabling the periodic boundary conditions in {vacuum_system_path} "
                f"by setting the cutoff type to NoCutoff"
            )

            with open(os.path.join(directory, vacuum_system_path), "r") as file:
                vacuum_system = XmlSerializer.deserialize(file.read())

            disable_pbc(vacuum_system)

            with open(os.path.join(directory, vacuum_system_path), "w") as file:
                file.write(XmlSerializer.serialize(vacuum_system))

        # Set up the yank input file.
        super(SolvationYankProtocol, self)._execute(directory, available_resources)

        if self.setup_only:
            return

        (
            self.solvent_1_free_energy,
            solution_1_trajectory,
            solvent_1_trajectory,
            solution_1_gradients,
            solvent_1_gradients,
        ) = self._analyze_phase(
            os.path.join(directory, "experiments", "solvent1.nc"),
            self.solution_1_system,
            "solvent1",
            available_resources,
        )

        self.solution_1_trajectory_path = os.path.join(directory, "solution_1.dcd")
        solution_1_trajectory.save_dcd(self.solution_1_trajectory_path)

        self.solvent_1_coordinate_path = os.path.join(directory, "solvent_1.pdb")
        self.solvent_1_trajectory_path = os.path.join(directory, "solvent_1.dcd")
        solvent_1_trajectory[0].save_pdb(self.solvent_1_coordinate_path)

        if solvent_1_trajectory.n_atoms > 0:
            solvent_1_trajectory.save_dcd(self.solvent_1_trajectory_path)
        else:
            with open(self.solvent_1_trajectory_path, "wb") as file:
                file.write(b"")

        (
            self.solvent_2_free_energy,
            solution_2_trajectory,
            solvent_2_trajectory,
            solution_2_gradients,
            solvent_2_gradients,
        ) = self._analyze_phase(
            os.path.join(directory, "experiments", "solvent2.nc"),
            self.solution_2_system,
            "solvent2",
            available_resources,
        )

        self.solution_2_trajectory_path = os.path.join(directory, "solution_2.dcd")
        solution_2_trajectory.save_dcd(self.solution_2_trajectory_path)

        self.solvent_2_coordinate_path = os.path.join(directory, "solvent_2.pdb")
        self.solvent_2_trajectory_path = os.path.join(directory, "solvent_2.dcd")
        solvent_2_trajectory[0].save_pdb(self.solvent_2_coordinate_path)

        if solvent_2_trajectory.n_atoms > 0:
            solvent_2_trajectory.save_dcd(self.solvent_2_trajectory_path)
        else:
            with open(self.solvent_2_trajectory_path, "wb") as file:
                file.write(b"")

        self.free_energy_difference = Observable(
            self.free_energy_difference.value.plus_minus(
                self.free_energy_difference.error
            ),
            gradients=[
                solution_1_gradients[key]
                - solvent_1_gradients[key]
                + solvent_2_gradients[key]
                - solution_2_gradients[key]
                for key in solvent_1_gradients
            ],
        )

        assert np.isclose(
            self.free_energy_difference.value,
            self.solvent_1_free_energy.value - self.solvent_2_free_energy.value,
        )
