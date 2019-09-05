"""
A collection of protocols for assigning force field parameters to molecular systems.
"""
import copy
import io
import logging
import re
import subprocess
from enum import Enum
from os import path

import numpy as np
import requests
from simtk import openmm
from simtk.openmm import app

from propertyestimator.forcefield import ForceFieldSource, SmirnoffForceFieldSource, LigParGenForceFieldSource, \
    TLeapForceFieldSource
from propertyestimator.substances import Substance
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.utils import temporarily_change_directory
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class BaseBuildSystemProtocol(BaseProtocol):
    """The base for any protocol whose role is to apply a set of
    force field parameters to a given system.
    """

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system.
        This path **must** point to a json serialized `SmirnoffForceFieldSource` object."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self, value):
        """The file path to the coordinate file which defines the system to which the
        force field parameters will be assigned."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system."""
        pass

    @protocol_output(str)
    def system_path(self):
        """The file path to the system object which contains the
        applied parameters."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new BaseBuildSystemProtocol object.
        """
        super().__init__(protocol_id)

        # Inputs
        self._force_field_path = None
        self._coordinate_file_path = None
        self._substance = None

        # Outputs
        self._system_path = None

    @staticmethod
    def _append_system(existing_system, system_to_append):
        """Appends a system object onto the end of an existing system.

        Parameters
        ----------
        existing_system: simtk.openmm.System
            The base system to extend.
        system_to_append: simtk.openmm.System
            The system to append.
        """
        supported_force_types = [
            openmm.HarmonicBondForce,
            openmm.HarmonicAngleForce,
            openmm.PeriodicTorsionForce,
            openmm.NonbondedForce,
        ]

        number_of_appended_forces = 0
        index_offset = existing_system.getNumParticles()

        # Append the particles.
        for index in range(system_to_append.getNumParticles()):
            existing_system.addParticle(system_to_append.getParticleMass(index))

        # Append the constraints
        for index in range(system_to_append.getNumConstraints()):

            index_a, index_b, distance = system_to_append.getConstraintParameters(index)
            existing_system.addConstraint(index_a + index_offset,
                                          index_b + index_offset, distance)

        # Append the forces.
        for existing_force in existing_system.getForces():

            if type(existing_force) not in supported_force_types:
                raise ValueError('The system contains an unsupported type of force.')

            for force_to_append in system_to_append.getForces():

                if type(force_to_append) != type(existing_force):
                    continue

                if isinstance(force_to_append, openmm.HarmonicBondForce):

                    # Add the bonds.
                    for index in range(force_to_append.getNumBonds()):

                        index_a, index_b, *parameters = force_to_append.getBondParameters(index)
                        existing_force.addBond(index_a + index_offset,
                                               index_b + index_offset, *parameters)

                elif isinstance(force_to_append, openmm.HarmonicAngleForce):

                    # Add the angles.
                    for index in range(force_to_append.getNumAngles()):

                        index_a, index_b, index_c, *parameters = force_to_append.getAngleParameters(index)
                        existing_force.addAngle(index_a + index_offset,
                                                index_b + index_offset,
                                                index_c + index_offset, *parameters)

                elif isinstance(force_to_append, openmm.PeriodicTorsionForce):

                    # Add the torsions.
                    for index in range(force_to_append.getNumTorsions()):

                        index_a, index_b, index_c, index_d, *parameters = force_to_append.getTorsionParameters(index)
                        existing_force.addTorsion(index_a + index_offset,
                                                  index_b + index_offset,
                                                  index_c + index_offset,
                                                  index_d + index_offset, *parameters)

                elif isinstance(force_to_append, openmm.NonbondedForce):

                    # Add the vdW parameters
                    for index in range(force_to_append.getNumParticles()):
                        existing_force.addParticle(*force_to_append.getParticleParameters(index))

                    # Add the 1-2, 1-3 and 1-4 exceptions.
                    for index in range(force_to_append.getNumExceptions()):

                        index_a, index_b, *parameters = force_to_append.getExceptionParameters(index)
                        existing_force.addException(index_a + index_offset,
                                                    index_b + index_offset, *parameters)

                number_of_appended_forces += 1

        if number_of_appended_forces != system_to_append.getNumForces():
            raise ValueError('Not all forces were appended.')

    def execute(self, directory, available_resources):
        raise NotImplementedError()


@register_calculation_protocol()
class BuildSmirnoffSystem(BaseBuildSystemProtocol):
    """Parametrise a set of molecules with a given smirnoff force field
    using the `OpenFF toolkit <https://github.com/openforcefield/openforcefield>`_.
    """
    class WaterModel(Enum):
        """An enum which describes which water model is being
        used, so that correct charges can be applied.

        Warnings
        --------
        This is only a temporary addition until library charges
        are introduced into the openforcefield toolkit.
        """
        TIP3P = 'TIP3P'

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system.
        This path **must** point to a json serialized `SmirnoffForceFieldSource` object."""
        pass

    @protocol_input(list)
    def charged_molecule_paths(self):
        """File paths to mol2 files which contain the charges assigned to molecules
        in the system. This input is helpful when dealing with large molecules (such
        as hosts in host-guest binding calculations) whose charges may by needed
        in multiple places, and hence should only be calculated once."""
        pass

    @protocol_input(WaterModel)
    def water_model(self):
        """The water model to apply, if any water molecules
        are present.

        Warnings
        --------
        This is only a temporary addition until library charges
        are introduced into the openforcefield toolkit.
        """
        pass

    @protocol_input(bool)
    def apply_known_charges(self):
        """If true, formal the formal charges of ions, and
        the charges of the selected water model will be
        automatically applied to any matching molecules in the
        system."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new `BuildSmirnoffSystem` object.
        """

        super().__init__(protocol_id)

        self._water_model = BuildSmirnoffSystem.WaterModel.TIP3P

        self._apply_known_charges = True
        self._charged_molecule_paths = []

    @staticmethod
    def _generate_known_charged_molecules():
        """Generates a set of molecules whose charges are known a priori,
        such as ions, for use in parameterised systems.

        Notes
        -----
        These are solely to be used as a work around until library charges
        are fully implemented in the openforcefield toolkit.

        Todos
        -----
        Remove this method when library charges are fully implemented in
        the openforcefield toolkit.

        Returns
        -------
        list of openforcefield.topology.Molecule
            The molecules with assigned charges.
        """
        from openforcefield.topology import Molecule
        from simtk import unit as simtk_unit

        sodium = Molecule.from_smiles('[Na+]')
        sodium.partial_charges = np.array([1.0]) * simtk_unit.elementary_charge

        potassium = Molecule.from_smiles('[K+]')
        potassium.partial_charges = np.array([1.0]) * simtk_unit.elementary_charge

        calcium = Molecule.from_smiles('[Ca+2]')
        calcium.partial_charges = np.array([2.0]) * simtk_unit.elementary_charge

        chlorine = Molecule.from_smiles('[Cl-]')
        chlorine.partial_charges = np.array([-1.0]) * simtk_unit.elementary_charge

        water = Molecule.from_smiles('O')
        water.partial_charges = np.array([-0.834, 0.417, 0.417]) * simtk_unit.elementary_charge

        return [sodium, potassium, calcium, chlorine, water]

    def execute(self, directory, available_resources):

        from openforcefield.topology import Molecule, Topology

        logging.info('Generating topology: ' + self.id)

        pdb_file = app.PDBFile(self._coordinate_file_path)

        try:

            with open(self._force_field_path) as file:
                force_field_source = ForceFieldSource.parse_json(file.read())

        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='{} could not load the ForceFieldSource: {}'.format(self.id, e))

        if not isinstance(force_field_source, SmirnoffForceFieldSource):

            return PropertyEstimatorException(directory=directory,
                                              message='Only SMIRNOFF force fields are supported by this '
                                                      'protocol.')

        force_field = force_field_source.to_force_field()

        unique_molecules = []
        charged_molecules = []

        if self._apply_known_charges:
            charged_molecules = self._generate_known_charged_molecules()

        # Load in any additional, user specified charged molecules.
        for charged_molecule_path in self._charged_molecule_paths:

            charged_molecule = Molecule.from_file(charged_molecule_path, 'MOL2')
            charged_molecules.append(charged_molecule)

        for component in self._substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)

            if molecule is None:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            unique_molecules.append(molecule)

        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)

        if len(charged_molecules) > 0:
            system = force_field.create_openmm_system(topology,
                                                      charge_from_molecules=charged_molecules)
        else:
            system = force_field.create_openmm_system(topology)

        if system is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Failed to create a system from the'
                                                       'provided topology and molecules')

        from simtk.openmm import XmlSerializer
        system_xml = XmlSerializer.serialize(system)

        self._system_path = path.join(directory, 'system.xml')

        with open(self._system_path, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

        logging.info('Topology generated: ' + self.id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class BuildLigParGenSystem(BaseBuildSystemProtocol):
    """Parametrise a set of molecules with the OPLS-AA/M force field.
    using the `LigParGen server <http://zarbi.chem.yale.edu/ligpargen/>`_.

    Notes
    -----
    This protocol is currently a work in progress and as such has limited
    functionality compared to the more established `BuildSmirnoffSystem` protocol.

    References
    ----------
    [1] Potential energy functions for atomic-level simulations of water and organic and
        biomolecular systems. Jorgensen, W. L.; Tirado-Rives, J. Proc. Nat. Acad. Sci.
        USA 2005, 102, 6665-6670
    [2] 1.14*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for Condensed-Phase
        Simulations. Dodda, L. S.; Vilseck, J. Z.; Tirado-Rives, J.; Jorgensen, W. L.
        J. Phys. Chem. B, 2017, 121 (15), pp 3864-3870
    [3] LigParGen web server: An automatic OPLS-AA parameter generator for organic ligands.
        Dodda, L. S.;Cabeza de Vaca, I.; Tirado-Rives, J.; Jorgensen, W. L.
        Nucleic Acids Research, Volume 45, Issue W1, 3 July 2017, Pages W331-W336
    """

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system.
        This path **must** point to a json serialized `LigParGenForceFieldSource` object."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new `BuildLigParGenSystem` object.
        """
        super().__init__(protocol_id)

    @staticmethod
    def _parameterize_smiles(smiles_pattern, force_field_source, directory):
        """Uses the `LigParGen` server to apply a set of parameters to
        a molecule defined by a smiles pattern.

        Parameters
        ----------
        smiles_pattern: str
            The smiles pattern which encodes the molecule to
            parametrize.
        force_field_source: LigParGenForceFieldSource
            The parameters to use in the parameterization.
        directory: str
            The directory to save the results in.

        Returns
        -------
        str
            A file path to the `simtk.openmm.app.ForceField` template.
        str
            A file path to the pdb file containing the coordinates and topology
            of the molecule.
        """
        from openforcefield.topology import Molecule

        initial_request_url = 'http://zarbi.chem.yale.edu/cgi-bin/results_lpg.py'
        empty_stream = io.BytesIO(b'\r\n')

        molecule = Molecule.from_smiles(smiles_pattern)
        total_charge = molecule.total_charge

        charge_model = 'cm1abcc'

        if (force_field_source.preferred_charge_model == LigParGenForceFieldSource.ChargeModel.CM1A_1_14 or
            not np.isclose(total_charge, 0.0)):

            charge_model = 'cm1a'

            if force_field_source.preferred_charge_model != LigParGenForceFieldSource.ChargeModel.CM1A_1_14:

                logging.warning(f'The preferred charge model is {str(force_field_source.preferred_charge_model)}, '
                                f'however the system is charged and so the '
                                f'{str(LigParGenForceFieldSource.ChargeModel.CM1A_1_14)} model will be used in its '
                                f'place.')

        data_body = {
            'smiData': (None, smiles_pattern),
            'molpdbfile': ('', empty_stream),
            'checkopt': (None, 0),
            'chargetype': (None, charge_model),
            'dropcharge': (None, total_charge)
        }

        # Perform the initial request for LigParGen to parameterize the molecule.
        request = requests.post(url=initial_request_url, files=data_body)

        # Cleanup the empty stream
        empty_stream.close()

        if request.status_code != requests.codes.ok:
            return f'The request failed with return code {request.status_code}.'

        response_content = request.content

        # Retrieve the server file name.
        force_field_file_name = re.search(r'value=\"\/tmp\/(.*?).xml\"', response_content.decode())

        if force_field_file_name is None:
            return 'The request could not successfully be completed.'

        force_field_file_name = force_field_file_name.group(1)

        # Download the force field xml file.
        download_request_url = 'http://zarbi.chem.yale.edu/cgi-bin/download_lpg.py'

        download_force_field_body = {
            'go': (None, 'XML'),
            'fileout': (None, f'/tmp/{force_field_file_name}.xml'),
        }

        request = requests.post(url=download_request_url, files=download_force_field_body)

        if request.status_code != requests.codes.ok:
            return f'The request to download the system xml file failed with return code {request.status_code}.'

        force_field_response = request.content
        force_field_path = path.join(directory, f'{smiles_pattern}.xml')

        with open(force_field_path, 'wb') as file:
            file.write(force_field_response)

        return force_field_path

    @staticmethod
    def _apply_opls_mixing_rules(system):
        """Applies the OPLS mixing rules to the system.

        Notes
        -----
        This method is based upon that found in the `LigParGen tutorial
        <http://zarbi.chem.yale.edu/ligpargen/openMM_tutorial.html>`_.

        Parameters
        ----------
        system: simtk.openmm.System
            The system object to apply the OPLS mixing rules to.
        """
        from simtk import unit as simtk_unit

        forces = [system.getForce(index) for index in range(system.getNumForces())]
        forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]

        for original_force in forces:

            # Define a custom force with the OPLS mixing rules.
            custom_force = openmm.CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); '
                                                       'sigma=sqrt(sigma1*sigma2); '
                                                       'epsilon=sqrt(epsilon1*epsilon2)')

            if original_force.getNonbondedMethod() == 4:  # Check for PME
                custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                custom_force.setNonbondedMethod(original_force.getNonbondedMethod())

            custom_force.addPerParticleParameter('sigma')
            custom_force.addPerParticleParameter('epsilon')
            custom_force.setCutoffDistance(original_force.getCutoffDistance())

            system.addForce(custom_force)

            lennard_jones_parameters = {}

            for index in range(original_force.getNumParticles()):
                charge, sigma, epsilon = original_force.getParticleParameters(index)

                # Copy the original vdW parameters over to the new custom force.
                lennard_jones_parameters[index] = (sigma, epsilon)
                custom_force.addParticle([sigma, epsilon])

                # Disable the original vdW interactions, but leave the charged interactions
                # turned on.
                original_force.setParticleParameters(index, charge, sigma, epsilon * 0)

            # Update the 1-4 exceptions.
            for exception_index in range(original_force.getNumExceptions()):

                (index_a, index_b, charge, sigma, epsilon) = original_force.getExceptionParameters(exception_index)

                # Disable any 1-2, 1-3, 1-4 exceptions on the custom force, and instead let the
                # original force handle it.
                custom_force.addExclusion(index_a, index_b)

                if not np.isclose(epsilon.value_in_unit(simtk_unit.kilojoule_per_mole), 0.0):
                    sigma_14 = np.sqrt(lennard_jones_parameters[index_a][0] *
                                       lennard_jones_parameters[index_b][0])

                    epsilon_14 = np.sqrt(lennard_jones_parameters[index_a][1] *
                                         lennard_jones_parameters[index_b][1])

                    original_force.setExceptionParameters(exception_index, index_a, index_b,
                                                          charge, sigma_14, epsilon_14)

    def execute(self, directory, available_resources):

        import mdtraj
        from openforcefield.topology import Molecule, Topology

        logging.info(f'Generating a system with LigParGen for {self._substance.identifier}: {self._id}')

        try:

            with open(self._force_field_path) as file:
                force_field_source = ForceFieldSource.parse_json(file.read())

        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='{} could not load the ForceFieldSource: {}'.format(self.id, e))

        if not isinstance(force_field_source, LigParGenForceFieldSource):

            return PropertyEstimatorException(directory=directory,
                                              message='Only SMIRNOFF force fields are supported by this '
                                                      'protocol.')

        # Load in the systems coordinates / topology
        openmm_pdb_file = app.PDBFile(self._coordinate_file_path)

        # Create an OFF topology for better insight into the layout of the system topology.
        unique_molecules = [Molecule.from_smiles(component.smiles) for
                            component in self._substance.components]

        # Create a dictionary of representative topology molecules for each component.
        topology = Topology.from_openmm(openmm_pdb_file.topology, unique_molecules)

        # Create the template system objects for each component in the system.
        system_templates = {}

        for index, component in enumerate(self._substance.components):

            # Create the force field template using the LigParGen server.
            force_field_path = self._parameterize_smiles(component.smiles, force_field_source, directory)

            reference_topology_molecule = None

            # Create temporary pdb files for each molecule type in the system, with their constituent
            # atoms ordered in the same way that they would be in the full system.
            for topology_molecule in topology.topology_molecules:

                if topology_molecule.reference_molecule.to_smiles() != unique_molecules[index].to_smiles():
                    continue

                reference_topology_molecule = topology_molecule
                break

            if reference_topology_molecule is None:
                return PropertyEstimatorException('A topology molecule could not be matched to its reference.')

            start_index = reference_topology_molecule.atom_start_topology_index
            end_index = start_index + reference_topology_molecule.n_atoms
            index_range = list(range(start_index, end_index))

            component_pdb_file = mdtraj.load_pdb(self._coordinate_file_path, atom_indices=index_range)
            component_topology = component_pdb_file.topology.to_openmm()
            component_topology.setUnitCellDimensions(openmm_pdb_file.topology.getUnitCellDimensions())

            # Create the system object.
            force_field_template = app.ForceField(force_field_path)

            component_system = force_field_template.createSystem(topology=component_topology,
                                                                 nonbondedMethod=app.PME,
                                                                 constraints=app.HBonds,
                                                                 removeCMMotion=False)

            system_templates[unique_molecules[index].to_smiles()] = component_system

        # Create the full system object from the component templates.
        system = None

        for topology_molecule in topology.topology_molecules:

            system_template = system_templates[topology_molecule.reference_molecule.to_smiles()]

            if system is None:

                # If no system has been set up yet, just use the first template.
                system = copy.deepcopy(system_template)
                continue

            # Append the component template to the full system.
            self._append_system(system, system_template)

        # Apply the OPLS mixing rules.
        self._apply_opls_mixing_rules(system)

        # Serialize the system object.
        system_xml = openmm.XmlSerializer.serialize(system)

        self._system_path = path.join(directory, 'system.xml')

        with open(self._system_path, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

        logging.info(f'System generated: {self.id}')

        return self._get_output_dictionary()


@register_calculation_protocol()
class BuildTLeapSystem(BaseBuildSystemProtocol):
    """Parametrise a set of molecules with an Amber based force field.
    using the `tleap package <http://ambermd.org/AmberTools.php>`_.

    Notes
    -----
    This protocol is currently a work in progress and as such has limited
    functionality compared to the more established `BuildSmirnoffSystem` protocol.
    """

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system.
        This path **must** point to a json serialized `TLeapForceFieldSource` object."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new `BuildTLeapSystem` object.
        """
        super().__init__(protocol_id)

    def topology_molecule_to_mol2(self, topology_molecule, file_name, toolkit='OE'):
        """Turn an openforcefield.topology.TopologyMolecule into a mol2 file, generating a conformer
           and charges in the process.

        .. warning :: This function uses non-public methods from the Open Force Field toolkit
                     and should be refactored when public methods become available

        .. note :: This function requires the OpenEye toolkit with a valid license to write mol2 format files

        Parameters
        ----------
        topology_molecule : openforcefield.topology.TopologyMolecule
            The TopologyMolecule to write out as a mol2 file. The atom ordering in this mol2 will
            be consistent with the topology ordering.
        file_name : string
            The filename to write to.
        toolkit : string. Allowed values are ['OE', 'RDKit', None]
            The cheminformatics toolkit to use for conformer generation and partial charge calculation.
            If None, geometries and partial charges from the underlying openforcefield.topology.Molecule will be used.
        """
        from simtk import unit
        import numpy as np
        from openforcefield.topology import Molecule
        from copy import deepcopy

        ALLOWED_TOOLKITS = ['openeye', 'rdkit', None]

        # Make a copy of the reference molecule so we can run conf gen / charge calc without modifying the original
        ref_mol = deepcopy(topology_molecule.reference_molecule)

        if toolkit is None:
            pass
        elif toolkit.lower() == 'openeye':
            from openforcefield.utils.toolkits import OpenEyeToolkitWrapper
            oetkw = OpenEyeToolkitWrapper()
            ref_mol.generate_conformers(toolkit_registry=oetkw)
            ref_mol.compute_partial_charges_am1bcc(toolkit_registry=oetkw)
        elif toolkit.lower() == 'rdkit':
            from openforcefield.utils.toolkits import RDKitToolkitWrapper, AmberToolsToolkitWrapper, ToolkitRegistry
            tkr = ToolkitRegistry(toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper])
            ref_mol.generate_conformers(toolkit_registry=tkr)
            ref_mol.compute_partial_charges_am1bcc(toolkit_registry=tkr)
        else:
            raise ValueError(f'Received invalid toolkit specification: {toolkit}. '
                             f'Allowed values are {ALLOWED_TOOLKITS}')


        # Get access to the parent topology, so we look up the topology atom indices later.
        topology = topology_molecule.topology

        # Make and populate a new openforcefield.topology.Molecule
        new_mol = Molecule()
        new_mol.name = ref_mol.name

        # Add atoms to the new molecule
        for top_atom in topology_molecule.atoms:

            # Force the topology to cache the topology molecule start indices
            topology.atom(top_atom.topology_atom_index)

            new_at_idx = new_mol.add_atom(top_atom.atom.atomic_number,
                                          top_atom.atom.formal_charge,
                                          top_atom.atom.is_aromatic,
                                          top_atom.atom.stereochemistry,
                                          top_atom.atom.name
                                          )

        # Add bonds to the new molecule
        for top_bond in topology_molecule.bonds:
            # This is a cheap hack to figure out what the "local" atom index of these atoms is.
            # In other words it is the offset we need to apply to get the index if this were
            # the only molecule in the whole Topology. We need to apply this offset because
            # new_mol begins its atom indexing at 0, not the real topology atom index (which we do know).
            idx_offset = topology_molecule._atom_start_topology_index

            # Convert the `.atoms` generator into a list so we can access it by index
            top_atoms = list(top_bond.atoms)

            new_bd_idx = new_mol.add_bond(top_atoms[0].topology_atom_index - idx_offset,
                                          top_atoms[1].topology_atom_index - idx_offset,
                                          top_bond.bond.bond_order,
                                          top_bond.bond.is_aromatic,
                                          top_bond.bond.stereochemistry,
                                          )

        # Transfer over existing conformers and partial charges, accounting for the
        # reference/topology indexing differences

        # We populate unitless arrays of the proper size and shape
        new_conf = np.zeros((ref_mol.n_atoms, 3))
        new_pcs = np.zeros(ref_mol.n_atoms)

        # Then iterate over the reference atoms, mapping their indices to the topology molecule's indexing system
        for ref_atom_idx in range(ref_mol.n_atoms):
            # We don't need to apply the offset here, since _ref_to_top_index is
            # already "locally" indexed for this topology molecule
            local_top_index = topology_molecule._ref_to_top_index[ref_atom_idx]

            # Strip the units becuase I'm lazy. We attach them below.
            new_conf[local_top_index, :] = ref_mol.conformers[0][ref_atom_idx] / unit.angstrom
            new_pcs[local_top_index] = ref_mol.partial_charges[ref_atom_idx] / unit.elementary_charge

        # Reattach the units
        new_mol.add_conformer(new_conf * unit.angstrom)
        new_mol.partial_charges = new_pcs * unit.elementary_charge

        # Write the molecule
        new_mol.to_file(file_name, file_format='mol2')


    def _run_tleap(self, force_field_source, initial_mol2_file_path, directory):
        """Uses tleap to apply parameters to a particular molecule,
        generating a `.prmtop` and a `.rst7` file with the applied parameters.

        Parameters
        ----------
        force_field_source: TLeapForceFieldSource
            The tleap source which describes which parameters to apply.
        smiles: str
            The MOL2 representation of the molecule to parameterise.
        directory: str
            The directory to store and temporary files / the final
            parameters in.

        Returns
        -------
        str
            The file path to the `prmtop` file.
        str
            The file path to the `rst7` file.
        PropertyEstimatorException, optional
            Any errors which were raised.
        """

        # Change into the working directory.
        with temporarily_change_directory(directory):

            amber_type = 'amber'

            if 'leaprc.gaff2' in force_field_source.leap_sources:
                amber_type = 'gaff2'
            elif 'leaprc.gaff' in force_field_source.leap_sources:
                amber_type = 'gaff'

            # Run antechamber to find the correct atom types.
            processed_mol2_path = 'antechamber.mol2'

            antechamber_result = subprocess.check_output(['antechamber',
                                                          '-i', initial_mol2_file_path, '-fi', 'mol2',
                                                          '-o', processed_mol2_path, '-fo', 'mol2',
                                                          '-at', amber_type,
                                                          '-rn', 'MOL',
                                                          '-an', 'no',
                                                          '-pf', 'yes'])

            with open('antechamber_output.log', 'w') as file:
                file.write(antechamber_result)

            if not path.isfile(processed_mol2_path):

                return None, None, PropertyEstimatorException(directory, f'antechamber failed to assign atom types to '
                                                                         f'the input mol2 file '
                                                                         f'({initial_mol2_file_path})')

            frcmod_path = None

            if amber_type == 'gaff' or amber_type == 'gaff2':

                # Optionally run parmchk to find any missing parameters.
                frcmod_path = 'parmck2.frcmod'

                prmchk2_result = subprocess.check_output(['parmchk2',
                                                          '-i', processed_mol2_path, '-f', 'mol2',
                                                          '-o', frcmod_path,
                                                          '-s', amber_type
                                                          ], cwd=directory)

                with open('parmchk2_output.log', 'w') as file:
                    file.write(prmchk2_result)

                if not path.isfile(frcmod_path):

                    return None, None, PropertyEstimatorException(directory,
                                                                  f'parmchk2 failed to assign missing {amber_type} '
                                                                  f'parameters to the antechamber created mol2 file '
                                                                  f'({processed_mol2_path})')

            # Build the tleap input file.
            template_lines = [f'source {source}' for source in force_field_source.leap_sources]

            if frcmod_path is not None:
                template_lines.append(f'loadamberparams {frcmod_path}', )

            prmtop_file_name = 'structure.prmtop'
            rst7_file_name = 'structure.rst7'

            template_lines.extend([
                f'MOL = loadmol2 {processed_mol2_path}',
                'check MOL',
                f'saveamberparm MOL {prmtop_file_name} {rst7_file_name}'
            ])

            input_file_path = 'tleap.in'

            with open(input_file_path, 'w') as file:
                file.write('\n'.join(template_lines))

            # Run tleap.
            tleap_result = subprocess.call(['tleap', '-s ', '-f ', input_file_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           cwd=directory)

            with open('tleap_output.log', 'w') as file:
                file.write(tleap_result)

            if not path.isfile(prmtop_file_name) or not path.isfile(rst7_file_name):
                return None, None, PropertyEstimatorException(directory, f'tleap failed to execute.')

            with open('leap.log', 'r') as file:

                if not re.search('ERROR|WARNING|Warning|duplicate|FATAL|Could|Fatal|Error', file.read()):
                    return path.join(directory, prmtop_file_name), path.join(directory, rst7_file_name), None

            return None, None, PropertyEstimatorException(directory, f'tleap failed to execute.')

    def execute(self, directory, available_resources):

        import mdtraj
        from openforcefield.topology import Molecule, Topology

        logging.info(f'Generating a system with tleap for {self._substance.identifier}: {self._id}')

        try:

            with open(self._force_field_path) as file:
                force_field_source = ForceFieldSource.parse_json(file.read())

        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='{} could not load the ForceFieldSource: {}'.format(self.id, e))

        if not isinstance(force_field_source, LigParGenForceFieldSource):

            return PropertyEstimatorException(directory=directory,
                                              message='Only SMIRNOFF force fields are supported by this '
                                                      'protocol.')

        # Load in the systems coordinates / topology
        openmm_pdb_file = app.PDBFile(self._coordinate_file_path)

        # Create an OFF topology for better insight into the layout of the system topology.
        unique_molecules = [Molecule.from_smiles(component.smiles) for
                            component in self._substance.components]

        topology = Topology.from_openmm(openmm_pdb_file.topology, unique_molecules)

        # Find a unique instance of each topology molecule to get the correct
        # atom orderings.
        topology_molecules = {}

        for topology_molecule in topology.topology_molecules:
            topology_molecules[topology_molecule.reference_molecule.to_smiles()] = topology_molecule

        #for smiles, topology_molecule in topology_molecules.items():


            # if reference_topology_molecule is None:
            #     return PropertyEstimatorException('A topology molecule could not be matched to its reference.')
            #
            # start_index = reference_topology_molecule.atom_start_topology_index
            # end_index = start_index + reference_topology_molecule.n_atoms
            # index_range = list(range(start_index, end_index))
            #
            # component_pdb_file = mdtraj.load_pdb(self._coordinate_file_path, atom_indices=index_range)
            # component_topology = component_pdb_file.topology.to_openmm()
            # component_topology.setUnitCellDimensions(openmm_pdb_file.topology.getUnitCellDimensions())
            #
            # # Create the system object.
            # force_field_template = app.ForceField(force_field_path)
            #
            # component_system = force_field_template.createSystem(topology=component_topology,
            #                                                      nonbondedMethod=app.PME,
            #                                                      constraints=app.HBonds,
            #                                                      removeCMMotion=False)

            # system_templates[unique_molecules[index].to_smiles()] = component_system

        # Create the full system object from the component templates.
        system = None

        for topology_molecule in topology.topology_molecules:

            system_template = system_templates[topology_molecule.reference_molecule.to_smiles()]

            if system is None:

                # If no system has been set up yet, just use the first template.
                system = copy.deepcopy(system_template)
                continue

            # Append the component template to the full system.
            self._append_system(system, system_template)

        # Apply the OPLS mixing rules.
        self._apply_opls_mixing_rules(system)

        # Serialize the system object.
        system_xml = openmm.XmlSerializer.serialize(system)

        self._system_path = path.join(directory, 'system.xml')

        with open(self._system_path, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

        logging.info(f'System generated: {self.id}')

        return self._get_output_dictionary()
