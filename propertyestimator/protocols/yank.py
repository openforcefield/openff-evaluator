"""
A collection of protocols for performing free energy calculations
using the YANK package.
"""
import logging
import os
import shutil
import threading
import traceback
from enum import Enum

import yaml

from propertyestimator import unit
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources, openmm_quantity_to_pint, \
    pint_quantity_to_openmm
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.utils import temporarily_change_directory
from propertyestimator.workflow.decorators import protocol_input, protocol_output, InequalityMergeBehaviour, UNDEFINED
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class BaseYankProtocol(BaseProtocol):
    """An abstract base class for protocols which will performs a set of alchemical
    free energy simulations using the YANK framework.

    Protocols which inherit from this base must implement the abstract `_get_yank_options`
    methods.
    """

    thermodynamic_state = protocol_input(
        docstring='The state at which to run the calculations.',
        type_hint=ThermodynamicState,
        default_value=UNDEFINED
    )

    number_of_iterations = protocol_input(
        docstring='The number of YANK iterations to perform.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=UNDEFINED
    )
    steps_per_iteration = protocol_input(
        docstring='The number of steps per YANK iteration to perform.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=500
    )
    checkpoint_interval = protocol_input(
        docstring='The number of iterations between saving YANK checkpoint files.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=50
    )

    timestep = protocol_input(
        docstring='The length of the timestep to take.',
        type_hint=unit.Quantity, merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2 * unit.femtosecond
    )

    force_field_path = protocol_input(
        docstring='The path to the force field to use for the calculations',
        type_hint=str,
        default_value=UNDEFINED
    )

    verbose = protocol_input(
        docstring='Controls whether or not to run YANK at high verbosity.',
        type_hint=bool,
        default_value=False
    )

    estimated_free_energy = protocol_output(
        docstring='The estimated free energy value and its uncertainty '
                  'returned by YANK.',
        type_hint=EstimatedQuantity
    )

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

        platform_name = 'CPU'

        if available_resources.number_of_gpus > 0:

            # A platform which runs on GPUs has been requested.
            from propertyestimator.backends import ComputeResources
            toolkit_enum = ComputeResources.GPUToolkit(available_resources.preferred_gpu_toolkit)

            # A platform which runs on GPUs has been requested.
            platform_name = 'CUDA' if toolkit_enum == ComputeResources.GPUToolkit.CUDA else \
                                                      ComputeResources.GPUToolkit.OpenCL

        return {
            'verbose': self.verbose,
            'output_dir': '.',

            'temperature': quantity_to_string(pint_quantity_to_openmm(self.thermodynamic_state.temperature)),
            'pressure': quantity_to_string(pint_quantity_to_openmm(self.thermodynamic_state.pressure)),

            'minimize': True,

            'default_number_of_iterations': self.number_of_iterations,
            'default_nsteps_per_iteration': self.steps_per_iteration,
            'checkpoint_interval': self.checkpoint_interval,

            'default_timestep': quantity_to_string(pint_quantity_to_openmm(self.timestep)),

            'annihilate_electrostatics': True,
            'annihilate_sterics': False,

            'platform': platform_name
        }

    def _get_solvent_dictionary(self):
        """Returns a dictionary of the solvent which will be serialized
        to a yaml file and passed to YANK. In most cases, this should
        just be passing force field settings over, such as PME settings.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK solvents.
        """
        from openforcefield.typing.engines.smirnoff.forcefield import ForceField
        force_field = ForceField(self.force_field_path)

        charge_method = force_field.get_parameter_handler('Electrostatics').method

        if charge_method.lower() != 'pme':
            raise ValueError('Currently only PME electrostatics are supported.')

        return {'default': {
            'nonbonded_method': charge_method,
        }}

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

        return {
            'system': system_key,
            'protocol': protocol_key
        }

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
            'options': self._get_options_dictionary(available_resources),

            'solvents': self._get_solvent_dictionary(),

            'systems': self._get_system_dictionary(),
            'protocols': self._get_protocol_dictionary(),

            'experiments': self._get_experiments_dictionary()
        }

    @staticmethod
    def _extract_trajectory(checkpoint_path, output_trajectory_path):
        """Extracts the stored trajectory of the 'initial' state from a
        yank `.nc` checkpoint file and stores it to disk as a `.dcd` file.

        Parameters
        ----------
        checkpoint_path: str
            The path to the yank `.nc` file
        output_trajectory_path: str
            The path to store the extracted trajectory at.
        """

        from yank.analyze import extract_trajectory

        mdtraj_trajectory = extract_trajectory(checkpoint_path, state_index=0, image_molecules=True)
        mdtraj_trajectory.save_dcd(output_trajectory_path)

    @staticmethod
    def _run_yank(directory, available_resources):
        """Runs YANK within the specified directory which contains a `yank.yaml`
        input file.

        Parameters
        ----------
        directory: str
            The directory within which to run yank.

        Returns
        -------
        simtk.unit.Quantity
            The free energy returned by yank.
        simtk.unit.Quantity
            The uncertainty in the free energy returned by yank.
        """

        from yank.experiment import ExperimentBuilder
        from yank.analyze import ExperimentAnalyzer

        with temporarily_change_directory(directory):

            # Set the default properties on the desired platform
            # before calling into yank.
            setup_platform_with_resources(available_resources)

            exp_builder = ExperimentBuilder('yank.yaml')
            exp_builder.run_experiments()

            analyzer = ExperimentAnalyzer('experiments')
            output = analyzer.auto_analyze()

            free_energy = output['free_energy']['free_energy_diff_unit']
            free_energy_uncertainty = output['free_energy']['free_energy_diff_error_unit']

        return free_energy, free_energy_uncertainty

    @staticmethod
    def _run_yank_as_process(queue, directory, available_resources):
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

        Returns
        -------
        simtk.unit.Quantity
            The free energy returned by yank.
        simtk.unit.Quantity
            The uncertainty in the free energy returned by yank.
        str, optional
            The stringified errors which occurred on the other process,
            or `None` if no exceptions were raised.
        """

        free_energy = None
        free_energy_uncertainty = None

        error = None

        try:
            free_energy, free_energy_uncertainty = BaseYankProtocol._run_yank(directory, available_resources)
        except Exception as e:
            error = traceback.format_exception(None, e, e.__traceback__)

        queue.put((free_energy, free_energy_uncertainty, error))

    def execute(self, directory, available_resources):

        yaml_filename = os.path.join(directory, 'yank.yaml')

        # Create the yank yaml input file from a dictionary of options.
        with open(yaml_filename, 'w') as file:
            yaml.dump(self._get_full_input_dictionary(available_resources), file)

        # Yank is not safe to be called from anything other than the main thread.
        # If the current thread is not detected as the main one, then yank should
        # be spun up in a new process which should itself be safe to run yank in.
        if threading.current_thread() is threading.main_thread():
            logging.info('Launching YANK in the main thread.')
            free_energy, free_energy_uncertainty = self._run_yank(directory, available_resources)
        else:

            from multiprocessing import Process, Queue

            logging.info('Launching YANK in a new process.')

            # Create a queue to pass the results back to the main process.
            queue = Queue()
            # Create the process within which yank will run.
            process = Process(target=BaseYankProtocol._run_yank_as_process, args=[queue, directory,
                                                                                  available_resources])

            # Start the process and gather back the output.
            process.start()
            free_energy, free_energy_uncertainty, error = queue.get()
            process.join()

            if error is not None:
                return PropertyEstimatorException(directory, error)

        self.estimated_free_energy = EstimatedQuantity(openmm_quantity_to_pint(free_energy),
                                                       openmm_quantity_to_pint(free_energy_uncertainty),
                                                       self._id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class LigandReceptorYankProtocol(BaseYankProtocol):
    """An abstract base class for protocols which will performs a set of
    alchemical free energy simulations using the YANK framework.

    Protocols which inherit from this base must implement the abstract
    `_get_*_dictionary` methods.
    """

    class RestraintType(Enum):
        """The types of ligand restraints available within yank.
        """
        Harmonic = 'Harmonic'
        FlatBottom = 'FlatBottom'

    ligand_residue_name = protocol_input(
        docstring='The residue name of the ligand.',
        type_hint=str,
        default_value=UNDEFINED
    )
    receptor_residue_name = protocol_input(
        docstring='The residue name of the receptor.',
        type_hint=str,
        default_value=UNDEFINED
    )

    solvated_ligand_coordinates = protocol_input(
        docstring='The file path to the solvated ligand coordinates.',
        type_hint=str,
        default_value=UNDEFINED
    )
    solvated_ligand_system = protocol_input(
        docstring='The file path to the solvated ligand system object.',
        type_hint=str,
        default_value=UNDEFINED
    )

    solvated_complex_coordinates = protocol_input(
        docstring='The file path to the solvated complex coordinates.',
        type_hint=str,
        default_value=UNDEFINED
    )
    solvated_complex_system = protocol_input(
        docstring='The file path to the solvated complex system object.',
        type_hint=str,
        default_value=UNDEFINED
    )

    apply_restraints = protocol_input(
        docstring='Determines whether the ligand should be explicitly restrained to the '
                  'receptor in order to stop the ligand from temporarily unbinding.',
        type_hint=bool,
        default_value=True
    )
    restraint_type = protocol_input(
        docstring='The type of ligand restraint applied, provided that `apply_restraints` '
                  'is `True`',
        type_hint=RestraintType,
        default_value=RestraintType.Harmonic
    )

    solvated_ligand_trajectory_path = protocol_output(
        docstring='The file path to the generated ligand trajectory.',
        type_hint=str
    )
    solvated_complex_trajectory_path = protocol_output(
        docstring='The file path to the generated ligand trajectory.',
        type_hint=str
    )

    def __init__(self, protocol_id):
        """Constructs a new LigandReceptorYankProtocol object."""

        super().__init__(protocol_id)

        self._local_ligand_coordinates = 'ligand.pdb'
        self._local_ligand_system = 'ligand.xml'

        self._local_complex_coordinates = 'complex.pdb'
        self._local_complex_system = 'complex.xml'

    def _get_system_dictionary(self):

        solvent_dictionary = self._get_solvent_dictionary()
        solvent_key = next(iter(solvent_dictionary))

        host_guest_dictionary = {
            'phase1_path': [self._local_complex_system, self._local_complex_coordinates],
            'phase2_path': [self._local_ligand_system, self._local_ligand_coordinates],

            'ligand_dsl': f'resname {self.ligand_residue_name}',
            'solvent': solvent_key
        }

        return {'host-guest': host_guest_dictionary}

    def _get_protocol_dictionary(self):

        absolute_binding_dictionary = {
            'complex': {'alchemical_path': 'auto'},
            'solvent': {'alchemical_path': 'auto'}
        }

        return {'absolute_binding_dictionary': absolute_binding_dictionary}

    def _get_experiments_dictionary(self):

        experiments_dictionary = super(LigandReceptorYankProtocol, self)._get_experiments_dictionary()

        if self.apply_restraints:

            experiments_dictionary['restraint'] = {
                'restrained_ligand_atoms': f'(resname {self.ligand_residue_name}) and (mass > 1.5)',
                'restrained_receptor_atoms': f'(resname {self.receptor_residue_name}) and (mass > 1.5)',

                'type': self.restraint_type.value
            }

        return experiments_dictionary

    def execute(self, directory, available_resources):

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(self.solvated_ligand_coordinates, os.path.join(directory, self._local_ligand_coordinates))
        shutil.copyfile(self.solvated_ligand_system, os.path.join(directory, self._local_ligand_system))

        shutil.copyfile(self.solvated_complex_coordinates, os.path.join(directory, self._local_complex_coordinates))
        shutil.copyfile(self.solvated_complex_system, os.path.join(directory, self._local_complex_system))

        result = super(LigandReceptorYankProtocol, self).execute(directory, available_resources)

        if isinstance(result, PropertyEstimatorException):
            return result

        ligand_yank_path = os.path.join(directory, 'experiments', 'solvent.nc')
        complex_yank_path = os.path.join(directory, 'experiments', 'complex.nc')

        self.solvated_ligand_trajectory_path = os.path.join(directory, 'ligand.dcd')
        self.solvated_complex_trajectory_path = os.path.join(directory, 'complex.dcd')

        self._extract_trajectory(ligand_yank_path, self.solvated_ligand_trajectory_path)
        self._extract_trajectory(complex_yank_path, self.solvated_complex_trajectory_path)

        return self._get_output_dictionary()
