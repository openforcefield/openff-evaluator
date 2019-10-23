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
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources, openmm_quantity_to_pint, \
    pint_quantity_to_openmm, disable_pbc
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

    number_of_equilibration_iterations = protocol_input(
        docstring='The number of iterations used for equilibration before production run. '
                  'Only post-equilibration iterations are written to file.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1
    )
    number_of_iterations = protocol_input(
        docstring='The number of YANK iterations to perform.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=5000
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

    verbose = protocol_input(
        docstring='Controls whether or not to run YANK at high verbosity.',
        type_hint=bool,
        default_value=False
    )
    setup_only = protocol_input(
        docstring='If true, YANK will only create and validate the setup files, '
                  'but not actually run any simulations. This argument is mainly '
                  'only to be used for testing purposes.',
        type_hint=bool,
        default_value=False
    )

    estimated_free_energy = protocol_output(
        docstring='The estimated free energy value and its uncertainty '
                  'returned by YANK.',
        type_hint=EstimatedQuantity
    )

    def _get_residue_names_from_role(self, role):
        """Returns a list of all of the residue names of
        components which have been assigned a given role.

        Parameters
        ----------
        role: Substance.ComponentRole
            The role of the component to identify.

        Returns
        -------
        set of str
            The identified residue names.
        """

        from simtk.openmm import app
        from openforcefield.topology import Molecule, Topology

        if role == Substance.ComponentRole.Undefined:
            return 'all'

        unique_molecules = [Molecule.from_smiles(component.smiles)
                            for component in self.substance.components]

        openmm_topology = app.PDBFile(self.solvated_coordinates).topology
        topology = Topology.from_openmm(openmm_topology, unique_molecules)

        # Determine the smiles of all molecules in the system. We need to use
        # the toolkit to re-generate the smiles as later we will compare these
        # against more toolkit generated smiles.
        components = [component for component in self.substance.components if
                      component.role == role]
        component_smiles = [Molecule.from_smiles(component.smiles).to_smiles() for
                            component in components]

        residue_names = set()

        all_openmm_atoms = list(openmm_topology.atoms())

        # Find the resiude names of the molecules which have the correct
        # role.
        for topology_molecule in topology.topology_molecules:

            molecule_smiles = topology_molecule.reference_molecule.to_smiles()

            if molecule_smiles not in component_smiles:
                continue

            molecule_residue_names = set([all_openmm_atoms[topology_atom.topology_atom_index].residue.name for
                                          topology_atom in topology_molecule.atoms])

            assert len(molecule_residue_names) == 1
            residue_names.update(molecule_residue_names)

        return residue_names

    def _get_dsl_from_role(self, role):
        """Returns an MDTraj DSL string which identifies those
        atoms which belong to components flagged with a specific
        role.

        Parameters
        ----------
        role: Substance.ComponentRole
            The role of the component to identify.

        Returns
        -------
        str
            The DSL string.
        """

        residue_names = self._get_residue_names_from_role(role)

        dsl_string = ' or '.join([f'resname {residue_name}' for residue_name in residue_names])
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

            'number_of_equilibration_iterations': self.number_of_equilibration_iterations,
            'default_number_of_iterations': self.number_of_iterations,
            'default_nsteps_per_iteration': self.steps_per_iteration,
            'checkpoint_interval': self.checkpoint_interval,

            'default_timestep': quantity_to_string(pint_quantity_to_openmm(self.timestep)),

            'annihilate_electrostatics': True,
            'annihilate_sterics': False,

            'platform': platform_name
        }

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
    def _run_yank(directory, available_resources, setup_only):
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
        simtk.unit.Quantity
            The free energy returned by yank.
        simtk.unit.Quantity
            The uncertainty in the free energy returned by yank.
        """

        from yank.experiment import ExperimentBuilder
        from yank.analyze import ExperimentAnalyzer

        from simtk import unit as simtk_unit

        with temporarily_change_directory(directory):

            # Set the default properties on the desired platform
            # before calling into yank.
            setup_platform_with_resources(available_resources)

            exp_builder = ExperimentBuilder('yank.yaml')

            if setup_only is True:
                return 0.0 * simtk_unit.kilojoule_per_mole, 0.0 * simtk_unit.kilojoule_per_mole

            exp_builder.run_experiments()

            analyzer = ExperimentAnalyzer('experiments')
            output = analyzer.auto_analyze()

            free_energy = output['free_energy']['free_energy_diff_unit']
            free_energy_uncertainty = output['free_energy']['free_energy_diff_error_unit']

        return free_energy, free_energy_uncertainty

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
            free_energy, free_energy_uncertainty = BaseYankProtocol._run_yank(directory,
                                                                              available_resources,
                                                                              setup_only)
        except Exception as e:
            error = traceback.format_exception(None, e, e.__traceback__)

        queue.put((free_energy, free_energy_uncertainty, error))

    def execute(self, directory, available_resources):

        yaml_filename = os.path.join(directory, 'yank.yaml')

        # Create the yank yaml input file from a dictionary of options.
        with open(yaml_filename, 'w') as file:
            yaml.dump(self._get_full_input_dictionary(available_resources), file, sort_keys=False)

        setup_only = self.setup_only

        # Yank is not safe to be called from anything other than the main thread.
        # If the current thread is not detected as the main one, then yank should
        # be spun up in a new process which should itself be safe to run yank in.
        if threading.current_thread() is threading.main_thread():
            logging.info('Launching YANK in the main thread.')
            free_energy, free_energy_uncertainty = self._run_yank(directory, available_resources, setup_only)
        else:

            from multiprocessing import Process, Queue

            logging.info('Launching YANK in a new process.')

            # Create a queue to pass the results back to the main process.
            queue = Queue()
            # Create the process within which yank will run.
            process = Process(target=BaseYankProtocol._run_yank_as_process, args=[queue, directory,
                                                                                  available_resources, setup_only])

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
    """A protocol for performing ligand-receptor alchemical free energy
    calculations using the YANK framework.
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

    force_field_path = protocol_input(
        docstring='The path to the force field which defines the charge method '
                  'to use for the calculation.',
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

    def _get_solvent_dictionary(self):
        """Returns a dictionary of the solvent which will be serialized
        to a yaml file and passed to YANK. In most cases, this should
        just be passing force field settings over, such as PME settings.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK solvents.
        """

        with open(self.force_field_path, 'r') as file:
            force_field_source = SmirnoffForceFieldSource.parse_json(file.read())

        force_field = force_field_source.to_force_field()
        charge_method = force_field.get_parameter_handler('Electrostatics').method

        if charge_method.lower() != 'pme':
            raise ValueError('Currently only PME electrostatics are supported.')

        return {'default': {
            'nonbonded_method': charge_method,
        }}

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

    def _get_full_input_dictionary(self, available_resources):

        full_dictionary = super(LigandReceptorYankProtocol, self)._get_full_input_dictionary(available_resources)
        full_dictionary['solvents'] = self._get_solvent_dictionary()

        return full_dictionary

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

        if self.setup_only:
            return self._get_output_dictionary()

        ligand_yank_path = os.path.join(directory, 'experiments', 'solvent.nc')
        complex_yank_path = os.path.join(directory, 'experiments', 'complex.nc')

        self.solvated_ligand_trajectory_path = os.path.join(directory, 'ligand.dcd')
        self.solvated_complex_trajectory_path = os.path.join(directory, 'complex.dcd')

        self._extract_trajectory(ligand_yank_path, self.solvated_ligand_trajectory_path)
        self._extract_trajectory(complex_yank_path, self.solvated_complex_trajectory_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class SolvationYankProtocol(BaseYankProtocol):
    """A protocol for performing solvation alchemical free energy
    calculations using the YANK framework.

    Todos
    -----
    * This protocol is could be easily generalized to facilitate
      transfer free energies, however I am currently unclear as to
      how yank determines what is solvent and what is solute in
      cases of different solvents in each phase.
    """

    substance = protocol_input(
        docstring='The substance describing the composition of '
                  'the system',
        type_hint=Substance,
        default_value=UNDEFINED
    )

    solvated_coordinates = protocol_input(
        docstring='The file path to the coordinates of the solute embedded in the '
                  'first solvent.',
        type_hint=str,
        default_value=UNDEFINED
    )
    solvated_system = protocol_input(
        docstring='The file path to the system object of the solute embedded in the '
                  'first solvent.',
        type_hint=str,
        default_value=UNDEFINED
    )

    vacuum_coordinates = protocol_input(
        docstring='The file path to the coordinates of the solute embedded in vacuum.',
        type_hint=str,
        default_value=UNDEFINED
    )
    vacuum_system = protocol_input(
        docstring='The file path to the system object of the solute embedded in vacuum.',
        type_hint=str,
        default_value=UNDEFINED
    )

    solvated_electrostatic_lambdas = protocol_input(
        docstring='The list of electrostatic alchemical states that YANK should sample at. '
                  'These values will be passed to the YANK `lambda_electrostatics` option. '
                  'If no option is set, YANK will use `trailblaze` algorithm to determine '
                  'this option automatically.',
        type_hint=list,
        optional=True,
        default_value=UNDEFINED
    )
    solvated_steric_lambdas = protocol_input(
        docstring='The list of steric alchemical states that YANK should sample at. '
                  'These values will be passed to the YANK `lambda_sterics` option. '
                  'If no option is set, YANK will use `trailblaze` algorithm to determine '
                  'this option automatically.',
        type_hint=list,
        optional=True,
        default_value=UNDEFINED
    )
    vacuum_electrostatic_lambdas = protocol_input(
        docstring='The list of electrostatic alchemical states that YANK should sample at. '
                  'These values will be passed to the YANK `lambda_electrostatics` option. '
                  'If no option is set, YANK will use `trailblaze` algorithm to determine '
                  'this option automatically.',
        type_hint=list,
        optional=True,
        default_value=UNDEFINED
    )
    vacuum_steric_lambdas = protocol_input(
        docstring='The list of steric alchemical states that YANK should sample at. '
                  'These values will be passed to the YANK `lambda_sterics` option. '
                  'If no option is set, YANK will use `trailblaze` algorithm to determine '
                  'this option automatically.',
        type_hint=list,
        optional=True,
        default_value=UNDEFINED
    )

    solvated_trajectory_path = protocol_output(
        docstring='The file path to the generated ligand trajectory.',
        type_hint=str
    )
    vacuum_trajectory_path = protocol_output(
        docstring='The file path to the generated ligand trajectory.',
        type_hint=str
    )

    def __init__(self, protocol_id):
        """Constructs a new LigandReceptorYankProtocol object."""

        super().__init__(protocol_id)

        self._local_solution_coordinates = 'solution.pdb'
        self._local_solution_system = 'solution.xml'

        self._local_vacuum_coordinates = 'vacuum.pdb'
        self._local_vacuum_system = 'vacuum.xml'

    def _get_system_dictionary(self):

        solvation_system_dictionary = {
            'phase1_path': [self._local_solution_system, self._local_solution_coordinates],
            'phase2_path': [self._local_vacuum_system, self._local_vacuum_coordinates],

            'solvent_dsl': self._get_dsl_from_role(Substance.ComponentRole.Solvent)
        }

        return {'solvation-system': solvation_system_dictionary}

    def _get_protocol_dictionary(self):

        solvent_1_protocol_dictionary = {
            'lambda_electrostatics': self.solvated_electrostatic_lambdas,
            'lambda_sterics': self.solvated_steric_lambdas
        }

        if (self.solvated_electrostatic_lambdas == UNDEFINED and
            self.solvated_steric_lambdas == UNDEFINED):

            solvent_1_protocol_dictionary = 'auto'

        elif ((self.solvated_electrostatic_lambdas != UNDEFINED and
               self.solvated_steric_lambdas == UNDEFINED) or
              (self.solvated_electrostatic_lambdas == UNDEFINED and
               self.solvated_steric_lambdas != UNDEFINED)):

            raise ValueError('Either both of `solvated_electrostatic_lambdas` and '
                             '`solvated_steric_lambdas` must be set, or neither '
                             'must be set.')

        solvent_2_protocol_dictionary = {
            'lambda_electrostatics': self.vacuum_electrostatic_lambdas,
            'lambda_sterics': self.vacuum_steric_lambdas
        }
        
        if (self.vacuum_electrostatic_lambdas == UNDEFINED and
            self.vacuum_steric_lambdas == UNDEFINED):

            solvent_2_protocol_dictionary = 'auto'

        elif ((self.vacuum_electrostatic_lambdas != UNDEFINED and
               self.vacuum_steric_lambdas == UNDEFINED) or
              (self.vacuum_electrostatic_lambdas == UNDEFINED and
               self.vacuum_steric_lambdas != UNDEFINED)):

            raise ValueError('Either both of `vacuum_electrostatic_lambdas` and '
                             '`vacuum_steric_lambdas` must be set, or neither '
                             'must be set.')

        absolute_binding_dictionary = {
            'solvent1': {'alchemical_path': solvent_1_protocol_dictionary},
            'solvent2': {'alchemical_path': solvent_2_protocol_dictionary}
        }

        return {'solvation-protocol': absolute_binding_dictionary}

    def execute(self, directory, available_resources):

        from simtk.openmm import XmlSerializer

        solutes = [component for component in self.substance.components if
                   component.role == Substance.ComponentRole.Solute]

        solvent = [component for component in self.substance.components if
                   component.role != Substance.ComponentRole.Solute]

        if len(solutes) != 1:

            return PropertyEstimatorException(directory, 'There must only be a single component marked as '
                                                         'a solute in the substance.')

        if len(solvent) <= 0:

            return PropertyEstimatorException(directory, 'There must be at least one solvent component in '
                                                         'the system.')

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(self.solvated_coordinates, os.path.join(directory, self._local_solution_coordinates))
        shutil.copyfile(self.solvated_system, os.path.join(directory, self._local_solution_system))

        shutil.copyfile(self.vacuum_coordinates, os.path.join(directory, self._local_vacuum_coordinates))
        shutil.copyfile(self.vacuum_system, os.path.join(directory, self._local_vacuum_system))

        # Disable any periodic boundary conditions on the vacuum system.
        logging.info(f'Disabling any PBC in {self._local_vacuum_system} by setting the '
                     f'cutoff type to NoCutoff')

        with open(os.path.join(directory, self._local_vacuum_system), 'r') as file:
            vacuum_system = XmlSerializer.deserialize(file.read())

        disable_pbc(vacuum_system)

        with open(os.path.join(directory, self._local_vacuum_system), 'w') as file:
            file.write(XmlSerializer.serialize(vacuum_system))

        result = super(SolvationYankProtocol, self).execute(directory, available_resources)

        if isinstance(result, PropertyEstimatorException):
            return result

        if self.setup_only:
            return self._get_output_dictionary()

        solvated_yank_path = os.path.join(directory, 'experiments', 'solvent1.nc')
        vacuum_yank_path = os.path.join(directory, 'experiments', 'solvent2.nc')

        self.solvated_trajectory_path = os.path.join(directory, 'solvated.dcd')
        self.vacuum_trajectory_path = os.path.join(directory, 'vacuum.dcd')

        self._extract_trajectory(solvated_yank_path, self.solvated_trajectory_path)
        self._extract_trajectory(vacuum_yank_path, self.vacuum_trajectory_path)

        return self._get_output_dictionary()
