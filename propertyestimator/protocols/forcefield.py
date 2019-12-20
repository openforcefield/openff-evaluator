"""
A collection of protocols for assigning force field parameters to molecular systems.
"""
import abc
import copy
import io
import logging
import os
import re
import shutil
import subprocess
from enum import Enum

import numpy as np
import requests
from simtk import openmm
from simtk.openmm import app

from propertyestimator.attributes import UNDEFINED
from propertyestimator.forcefield import (
    ForceFieldSource,
    LigParGenForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from propertyestimator.substances import Substance
from propertyestimator.utils.openmm import pint_quantity_to_openmm
from propertyestimator.utils.utils import (
    get_data_filename,
    temporarily_change_directory,
)
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import WorkflowProtocol


class BuildSystemProtocol(WorkflowProtocol, abc.ABC):
    """The base for any protocol whose role is to apply a set of
    force field parameters to a given system.
    """

    class WaterModel(Enum):
        """An enum which describes which water model is being
        used, so that correct charges can be applied.

        Warnings
        --------
        This is only a temporary addition until full water model support
        is introduced.
        """

        TIP3P = "TIP3P"

    force_field_path = InputAttribute(
        docstring="The file path to the force field parameters to assign to the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    coordinate_file_path = InputAttribute(
        docstring="The file path to the PDB coordinate file which defines the "
        "topology of the system to which the force field parameters "
        "will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    substance = InputAttribute(
        docstring="The composition of the system.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    water_model = InputAttribute(
        docstring="The water model to apply, if any water molecules are present.",
        type_hint=WaterModel,
        default_value=WaterModel.TIP3P,
    )

    system_path = OutputAttribute(
        docstring="The path to the assigned system object.", type_hint=str
    )

    @staticmethod
    def _build_tip3p_system(topology_molecule, cutoff, cell_vectors):
        """Builds a `simtk.openmm.System` object containing a single water model

        Parameters
        ----------
        topology_molecule: openforcefield.topology.TopologyMolecule
            The topology molecule which represents the water molecule
            in the full system.
        cutoff: simtk.unit.Quantity
            The non-bonded cutoff.
        cell_vectors: simtk.unit.Quantity
            The full system's cell vectors.

        Returns
        -------
        simtk.openmm.System
            The created system.
        """

        topology_atoms = list(topology_molecule.atoms)

        # Make sure the topology molecule is in the order we expect.
        assert len(topology_atoms) == 3

        assert topology_atoms[0].atom.element.symbol == "O"
        assert topology_atoms[1].atom.element.symbol == "H"
        assert topology_atoms[2].atom.element.symbol == "H"

        force_field_path = get_data_filename("forcefield/tip3p.xml")
        water_pdb_path = get_data_filename("forcefield/tip3p.pdb")

        component_pdb_file = app.PDBFile(water_pdb_path)
        component_topology = component_pdb_file.topology
        component_topology.setUnitCellDimensions(cell_vectors)

        # Create the system object.
        force_field_template = app.ForceField(force_field_path)

        component_system = force_field_template.createSystem(
            topology=component_topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
        )

        return component_system

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
            existing_system.addConstraint(
                index_a + index_offset, index_b + index_offset, distance
            )

        # Validate the forces to append.
        for force_to_append in system_to_append.getForces():

            if type(force_to_append) in supported_force_types:
                continue

            raise ValueError(
                f"The system contains an unsupported type of "
                f"force: {type(force_to_append)}."
            )

        # Append the forces.
        for force_to_append in system_to_append.getForces():

            existing_force = None

            for force in existing_system.getForces():

                if type(force) not in supported_force_types:

                    raise ValueError(
                        f"The existing system contains an unsupported type "
                        f"of force: {type(force)}."
                    )

                if type(force_to_append) != type(force):
                    continue

                existing_force = force
                break

            if existing_force is None:

                existing_force = type(force_to_append)()
                existing_system.addForce(existing_force)

            if isinstance(force_to_append, openmm.HarmonicBondForce):

                # Add the bonds.
                for index in range(force_to_append.getNumBonds()):

                    index_a, index_b, *parameters = force_to_append.getBondParameters(
                        index
                    )
                    existing_force.addBond(
                        index_a + index_offset, index_b + index_offset, *parameters
                    )

            elif isinstance(force_to_append, openmm.HarmonicAngleForce):

                # Add the angles.
                for index in range(force_to_append.getNumAngles()):

                    (
                        index_a,
                        index_b,
                        index_c,
                        *parameters,
                    ) = force_to_append.getAngleParameters(index)
                    existing_force.addAngle(
                        index_a + index_offset,
                        index_b + index_offset,
                        index_c + index_offset,
                        *parameters,
                    )

            elif isinstance(force_to_append, openmm.PeriodicTorsionForce):

                # Add the torsions.
                for index in range(force_to_append.getNumTorsions()):

                    (
                        index_a,
                        index_b,
                        index_c,
                        index_d,
                        *parameters,
                    ) = force_to_append.getTorsionParameters(index)
                    existing_force.addTorsion(
                        index_a + index_offset,
                        index_b + index_offset,
                        index_c + index_offset,
                        index_d + index_offset,
                        *parameters,
                    )

            elif isinstance(force_to_append, openmm.NonbondedForce):

                # Add the vdW parameters
                for index in range(force_to_append.getNumParticles()):
                    existing_force.addParticle(
                        *force_to_append.getParticleParameters(index)
                    )

                # Add the 1-2, 1-3 and 1-4 exceptions.
                for index in range(force_to_append.getNumExceptions()):

                    (
                        index_a,
                        index_b,
                        *parameters,
                    ) = force_to_append.getExceptionParameters(index)
                    existing_force.addException(
                        index_a + index_offset, index_b + index_offset, *parameters
                    )

            number_of_appended_forces += 1

        if number_of_appended_forces != system_to_append.getNumForces():
            raise ValueError("Not all forces were appended.")


@workflow_protocol()
class BuildSmirnoffSystem(BuildSystemProtocol):
    """Parametrise a set of molecules with a given smirnoff force field
    using the `OpenFF toolkit <https://github.com/openforcefield/openforcefield>`_.
    """

    charged_molecule_paths = InputAttribute(
        docstring="File paths to mol2 files which contain the charges assigned to "
        "molecules in the system. This input is helpful when dealing "
        "with large molecules (such as hosts in host-guest binding "
        "calculations) whose charges may by needed in multiple places,"
        " and hence should only be calculated once.",
        type_hint=list,
        default_value=[],
    )
    apply_known_charges = InputAttribute(
        docstring="If true, the formal charges of ions and the partial charges of "
        "the selected water model will be automatically applied to any "
        "matching molecules in the system.",
        type_hint=bool,
        default_value=True,
    )

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

        sodium = Molecule.from_smiles("[Na+]")
        sodium.partial_charges = np.array([1.0]) * simtk_unit.elementary_charge

        potassium = Molecule.from_smiles("[K+]")
        potassium.partial_charges = np.array([1.0]) * simtk_unit.elementary_charge

        calcium = Molecule.from_smiles("[Ca+2]")
        calcium.partial_charges = np.array([2.0]) * simtk_unit.elementary_charge

        chlorine = Molecule.from_smiles("[Cl-]")
        chlorine.partial_charges = np.array([-1.0]) * simtk_unit.elementary_charge

        water = Molecule.from_smiles("O")
        water.partial_charges = (
            np.array([-0.834, 0.417, 0.417]) * simtk_unit.elementary_charge
        )

        return [sodium, potassium, calcium, chlorine, water]

    def execute(self, directory, available_resources):

        from openforcefield.topology import Molecule, Topology

        pdb_file = app.PDBFile(self.coordinate_file_path)

        with open(self.force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, SmirnoffForceFieldSource):
            raise ValueError(
                "Only SMIRNOFF force fields are supported by this protocol."
            )

        force_field = force_field_source.to_force_field()

        unique_molecules = []
        charged_molecules = []

        if self.apply_known_charges:
            charged_molecules = self._generate_known_charged_molecules()

        # Load in any additional, user specified charged molecules.
        for charged_molecule_path in self.charged_molecule_paths:

            charged_molecule = Molecule.from_file(charged_molecule_path, "MOL2")
            charged_molecules.append(charged_molecule)

        for component in self.substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)

            if molecule is None:
                raise ValueError(f"{component} could not be converted to a Molecule")

            unique_molecules.append(molecule)

        topology = Topology.from_openmm(
            pdb_file.topology, unique_molecules=unique_molecules
        )

        if len(charged_molecules) > 0:
            system = force_field.create_openmm_system(
                topology, charge_from_molecules=charged_molecules
            )
        else:
            system = force_field.create_openmm_system(topology)

        if system is None:

            raise RuntimeError(
                "Failed to create a system from the specified topology and molecules."
            )

        system_xml = openmm.XmlSerializer.serialize(system)
        self.system_path = os.path.join(directory, "system.xml")

        with open(self.system_path, "w") as file:
            file.write(system_xml)

        return self._get_output_dictionary()


@workflow_protocol()
class BuildLigParGenSystem(BuildSystemProtocol):
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

        initial_request_url = force_field_source.request_url
        empty_stream = io.BytesIO(b"\r\n")

        molecule = Molecule.from_smiles(smiles_pattern)
        total_charge = molecule.total_charge

        charge_model = "cm1abcc"

        if (
            force_field_source.preferred_charge_model
            == LigParGenForceFieldSource.ChargeModel.CM1A_1_14
            or not np.isclose(total_charge, 0.0)
        ):

            charge_model = "cm1a"

            if (
                force_field_source.preferred_charge_model
                != LigParGenForceFieldSource.ChargeModel.CM1A_1_14
            ):

                logging.warning(
                    f"The preferred charge model is {str(force_field_source.preferred_charge_model)}, "
                    f"however the system is charged and so the "
                    f"{str(LigParGenForceFieldSource.ChargeModel.CM1A_1_14)} model will be used in its "
                    f"place."
                )

        data_body = {
            "smiData": (None, smiles_pattern),
            "molpdbfile": ("", empty_stream),
            "checkopt": (None, 0),
            "chargetype": (None, charge_model),
            "dropcharge": (None, total_charge),
        }

        # Perform the initial request for LigParGen to parameterize the molecule.
        request = requests.post(url=initial_request_url, files=data_body)

        # Cleanup the empty stream
        empty_stream.close()

        if request.status_code != requests.codes.ok:
            return f"The request failed with return code {request.status_code}."

        response_content = request.content

        # Retrieve the server file name.
        force_field_file_name = re.search(
            r"value=\"/tmp/(.*?).xml\"", response_content.decode()
        )

        if force_field_file_name is None:
            return "The request could not successfully be completed."

        force_field_file_name = force_field_file_name.group(1)

        # Download the force field xml file.
        download_request_url = force_field_source.download_url

        download_force_field_body = {
            "go": (None, "XML"),
            "fileout": (None, f"/tmp/{force_field_file_name}.xml"),
        }

        request = requests.post(
            url=download_request_url, files=download_force_field_body
        )

        if request.status_code != requests.codes.ok:
            return f"The request to download the system xml file failed with return code {request.status_code}."

        force_field_response = request.content
        force_field_path = os.path.join(directory, f"{smiles_pattern}.xml")

        with open(force_field_path, "wb") as file:
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
            custom_force = openmm.CustomNonbondedForce(
                "4*epsilon*((sigma/r)^12-(sigma/r)^6); "
                "sigma=sqrt(sigma1*sigma2); "
                "epsilon=sqrt(epsilon1*epsilon2)"
            )

            if original_force.getNonbondedMethod() == 4:  # Check for PME
                custom_force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.CutoffPeriodic
                )
            else:
                custom_force.setNonbondedMethod(original_force.getNonbondedMethod())

            custom_force.addPerParticleParameter("sigma")
            custom_force.addPerParticleParameter("epsilon")
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

                (
                    index_a,
                    index_b,
                    charge,
                    sigma,
                    epsilon,
                ) = original_force.getExceptionParameters(exception_index)

                # Disable any 1-2, 1-3, 1-4 exceptions on the custom force, and instead let the
                # original force handle it.
                custom_force.addExclusion(index_a, index_b)

                if not np.isclose(
                    epsilon.value_in_unit(simtk_unit.kilojoule_per_mole), 0.0
                ):
                    sigma_14 = np.sqrt(
                        lennard_jones_parameters[index_a][0]
                        * lennard_jones_parameters[index_b][0]
                    )

                    epsilon_14 = np.sqrt(
                        lennard_jones_parameters[index_a][1]
                        * lennard_jones_parameters[index_b][1]
                    )

                    original_force.setExceptionParameters(
                        exception_index, index_a, index_b, charge, sigma_14, epsilon_14
                    )

    def execute(self, directory, available_resources):

        import mdtraj
        from openforcefield.topology import Molecule, Topology

        with open(self.force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, LigParGenForceFieldSource):

            raise ValueError(
                "Only LigParGen force field sources are supported by this protocol."
            )

        # Load in the systems coordinates / topology
        openmm_pdb_file = app.PDBFile(self.coordinate_file_path)

        # Create an OFF topology for better insight into the layout of the system topology.
        unique_molecules = [
            Molecule.from_smiles(component.smiles)
            for component in self.substance.components
        ]

        # Create a dictionary of representative topology molecules for each component.
        topology = Topology.from_openmm(openmm_pdb_file.topology, unique_molecules)

        # Create the template system objects for each component in the system.
        system_templates = {}

        cutoff = pint_quantity_to_openmm(force_field_source.cutoff)

        for index, component in enumerate(self.substance.components):

            reference_topology_molecule = None

            # Create temporary pdb files for each molecule type in the system, with their constituent
            # atoms ordered in the same way that they would be in the full system.
            topology_molecule = None

            for topology_molecule in topology.topology_molecules:

                if (
                    topology_molecule.reference_molecule.to_smiles()
                    != unique_molecules[index].to_smiles()
                ):
                    continue

                reference_topology_molecule = topology_molecule
                break

            if reference_topology_molecule is None or topology_molecule is None:

                raise ValueError(
                    "A topology molecule could not be matched to its reference."
                )

            # Create the force field template using the LigParGen server.
            if component.smiles != "O" and component.smiles != "[H]O[H]":

                force_field_path = self._parameterize_smiles(
                    component.smiles, force_field_source, directory
                )

                start_index = reference_topology_molecule.atom_start_topology_index
                end_index = start_index + reference_topology_molecule.n_atoms
                index_range = list(range(start_index, end_index))

                component_pdb_file = mdtraj.load_pdb(
                    self.coordinate_file_path, atom_indices=index_range
                )
                component_topology = component_pdb_file.topology.to_openmm()
                component_topology.setUnitCellDimensions(
                    openmm_pdb_file.topology.getUnitCellDimensions()
                )

                # Create the system object.
                # noinspection PyTypeChecker
                force_field_template = app.ForceField(force_field_path)

                component_system = force_field_template.createSystem(
                    topology=component_topology,
                    nonbondedMethod=app.PME,
                    nonbondedCutoff=cutoff,
                    constraints=app.HBonds,
                    rigidWater=True,
                    removeCMMotion=False,
                )
            else:

                component_system = self._build_tip3p_system(
                    topology_molecule,
                    cutoff,
                    openmm_pdb_file.topology.getUnitCellDimensions(),
                )

            system_templates[unique_molecules[index].to_smiles()] = component_system

        # Create the full system object from the component templates.
        system = None

        for topology_molecule in topology.topology_molecules:

            system_template = system_templates[
                topology_molecule.reference_molecule.to_smiles()
            ]

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

        self.system_path = os.path.join(directory, "system.xml")

        with open(self.system_path, "w") as file:
            file.write(system_xml)

        return self._get_output_dictionary()


@workflow_protocol()
class BuildTLeapSystem(BuildSystemProtocol):
    """Parametrise a set of molecules with an Amber based force field.
    using the `tleap package <http://ambermd.org/AmberTools.php>`_.

    Notes
    -----
    * This protocol is currently a work in progress and as such has limited
      functionality compared to the more established `BuildSmirnoffSystem` protocol.
    * This protocol requires the optional `ambertools ==19.0` dependency to be installed.
    """

    class ChargeBackend(Enum):
        """The framework to use to assign partial charges.
        """

        OpenEye = "OpenEye"
        AmberTools = "AmberTools"

    charge_backend = InputAttribute(
        docstring="The backend framework to use to assign partial charges.",
        type_hint=ChargeBackend,
        default_value=ChargeBackend.OpenEye,
    )

    @staticmethod
    def _topology_molecule_to_mol2(topology_molecule, file_name, charge_backend):
        """Converts an `openforcefield.topology.TopologyMolecule` into a mol2 file,
        generating a conformer and AM1BCC charges in the process.

        .. todo :: This function uses non-public methods from the Open Force Field toolkit
                   and should be refactored when public methods become available

        Parameters
        ----------
        topology_molecule: openforcefield.topology.TopologyMolecule
            The `TopologyMolecule` to write out as a mol2 file. The atom ordering in
            this mol2 will be consistent with the topology ordering.
        file_name: str
            The filename to write to.
        charge_backend: BuildTLeapSystem.ChargeBackend
            The backend to use for conformer generation and partial charge
            calculation.
        """
        from openforcefield.topology import Molecule
        from simtk import unit as simtk_unit

        # Make a copy of the reference molecule so we can run conf gen / charge calc without modifying the original
        reference_molecule = copy.deepcopy(topology_molecule.reference_molecule)

        if charge_backend == BuildTLeapSystem.ChargeBackend.OpenEye:

            from openforcefield.utils.toolkits import OpenEyeToolkitWrapper

            toolkit_wrapper = OpenEyeToolkitWrapper()
            reference_molecule.generate_conformers(toolkit_registry=toolkit_wrapper)
            reference_molecule.compute_partial_charges_am1bcc(
                toolkit_registry=toolkit_wrapper
            )

        elif charge_backend == BuildTLeapSystem.ChargeBackend.AmberTools:

            from openforcefield.utils.toolkits import (
                RDKitToolkitWrapper,
                AmberToolsToolkitWrapper,
                ToolkitRegistry,
            )

            toolkit_wrapper = ToolkitRegistry(
                toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
            )
            reference_molecule.generate_conformers(toolkit_registry=toolkit_wrapper)
            reference_molecule.compute_partial_charges_am1bcc(
                toolkit_registry=toolkit_wrapper
            )

        else:
            raise ValueError(f"Invalid toolkit specification.")

        # Get access to the parent topology, so we can look up the topology atom indices later.
        topology = topology_molecule.topology

        # Make and populate a new openforcefield.topology.Molecule
        new_molecule = Molecule()
        new_molecule.name = reference_molecule.name

        # Add atoms to the new molecule in the correct order
        for topology_atom in topology_molecule.atoms:

            # Force the topology to cache the topology molecule start indices
            topology.atom(topology_atom.topology_atom_index)

            new_molecule.add_atom(
                topology_atom.atom.atomic_number,
                topology_atom.atom.formal_charge,
                topology_atom.atom.is_aromatic,
                topology_atom.atom.stereochemistry,
                topology_atom.atom.name,
            )

        # Add bonds to the new molecule
        for topology_bond in topology_molecule.bonds:

            # This is a temporary workaround to figure out what the "local" atom index of
            # these atoms is. In other words it is the offset we need to apply to get the
            # index if this were the only molecule in the whole Topology. We need to apply
            # this offset because `new_molecule` begins its atom indexing at 0, not the
            # real topology atom index (which we do know).
            # noinspection PyProtectedMember
            index_offset = topology_molecule._atom_start_topology_index

            # Convert the `.atoms` generator into a list so we can access it by index
            topology_atoms = list(topology_bond.atoms)

            new_molecule.add_bond(
                topology_atoms[0].topology_atom_index - index_offset,
                topology_atoms[1].topology_atom_index - index_offset,
                topology_bond.bond.bond_order,
                topology_bond.bond.is_aromatic,
                topology_bond.bond.stereochemistry,
            )

        # Transfer over existing conformers and partial charges, accounting for the
        # reference/topology indexing differences
        new_conformers = np.zeros((reference_molecule.n_atoms, 3))
        new_charges = np.zeros(reference_molecule.n_atoms)

        # Then iterate over the reference atoms, mapping their indices to the topology
        # molecule's indexing system
        for reference_atom_index in range(reference_molecule.n_atoms):
            # We don't need to apply the offset here, since _ref_to_top_index is
            # already "locally" indexed for this topology molecule
            # noinspection PyProtectedMember
            local_top_index = topology_molecule._ref_to_top_index[reference_atom_index]

            new_conformers[local_top_index, :] = reference_molecule.conformers[0][
                reference_atom_index
            ].value_in_unit(simtk_unit.angstrom)
            new_charges[local_top_index] = reference_molecule.partial_charges[
                reference_atom_index
            ].value_in_unit(simtk_unit.elementary_charge)

        # Reattach the units
        new_molecule.add_conformer(new_conformers * simtk_unit.angstrom)
        new_molecule.partial_charges = new_charges * simtk_unit.elementary_charge

        # Write the molecule
        new_molecule.to_file(file_name, file_format="mol2")

    @staticmethod
    def _run_tleap(force_field_source, initial_mol2_file_path, directory):
        """Uses tleap to apply parameters to a particular molecule,
        generating a `.prmtop` and a `.rst7` file with the applied parameters.

        Parameters
        ----------
        force_field_source: TLeapForceFieldSource
            The tleap source which describes which parameters to apply.
        initial_mol2_file_path: str
            The path to the MOL2 representation of the molecule to parameterize.
        directory: str
            The directory to store and temporary files / the final
            parameters in.

        Returns
        -------
        str
            The file path to the `prmtop` file.
        str
            The file path to the `rst7` file.
        """

        # Change into the working directory.
        with temporarily_change_directory(directory):

            if force_field_source.leap_source == "leaprc.gaff2":
                amber_type = "gaff2"
            elif force_field_source.leap_source == "leaprc.gaff":
                amber_type = "gaff"
            else:

                raise ValueError(
                    f"The {force_field_source.leap_source} source is currently "
                    f"unsupported. Only the 'leaprc.gaff2' and 'leaprc.gaff' "
                    f" sources are supported."
                )

            # Run antechamber to find the correct atom types.
            processed_mol2_path = "antechamber.mol2"

            antechamber_process = subprocess.Popen(
                [
                    "antechamber",
                    "-i",
                    initial_mol2_file_path,
                    "-fi",
                    "mol2",
                    "-o",
                    processed_mol2_path,
                    "-fo",
                    "mol2",
                    "-at",
                    amber_type,
                    "-rn",
                    "MOL",
                    "-an",
                    "no",
                    "-pf",
                    "yes",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            antechamber_output, antechamber_error = antechamber_process.communicate()
            antechamber_exit_code = antechamber_process.returncode

            with open("antechamber_output.log", "w") as file:
                file.write(f"error code: {antechamber_exit_code}\nstdout:\n\n")
                file.write("stdout:\n\n")
                file.write(antechamber_output.decode())
                file.write("\nstderr:\n\n")
                file.write(antechamber_error.decode())

            if not os.path.isfile(processed_mol2_path):

                raise RuntimeError(
                    f"antechamber failed to assign atom types to the input mol2 file "
                    f"({initial_mol2_file_path})"
                )

            frcmod_path = None

            if amber_type == "gaff" or amber_type == "gaff2":

                # Optionally run parmchk to find any missing parameters.
                frcmod_path = "parmck2.frcmod"

                prmchk2_process = subprocess.Popen(
                    [
                        "parmchk2",
                        "-i",
                        processed_mol2_path,
                        "-f",
                        "mol2",
                        "-o",
                        frcmod_path,
                        "-s",
                        amber_type,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                prmchk2_output, prmchk2_error = prmchk2_process.communicate()
                prmchk2_exit_code = prmchk2_process.returncode

                with open("prmchk2_output.log", "w") as file:
                    file.write(f"error code: {prmchk2_exit_code}\nstdout:\n\n")
                    file.write(prmchk2_output.decode())
                    file.write("\nstderr:\n\n")
                    file.write(prmchk2_error.decode())

                if not os.path.isfile(frcmod_path):

                    raise RuntimeError(
                        f"parmchk2 failed to assign missing {amber_type} parameters "
                        f"to the antechamber created mol2 file ({processed_mol2_path})",
                    )

            # Build the tleap input file.
            template_lines = [f"source {force_field_source.leap_source}"]

            if frcmod_path is not None:
                template_lines.append(f"loadamberparams {frcmod_path}",)

            prmtop_file_name = "structure.prmtop"
            rst7_file_name = "structure.rst7"

            template_lines.extend(
                [
                    f"MOL = loadmol2 {processed_mol2_path}",
                    f'setBox MOL "centers"',
                    "check MOL",
                    f"saveamberparm MOL {prmtop_file_name} {rst7_file_name}",
                ]
            )

            input_file_path = "tleap.in"

            with open(input_file_path, "w") as file:
                file.write("\n".join(template_lines))

            # Run tleap.
            tleap_process = subprocess.Popen(
                ["tleap", "-s ", "-f ", input_file_path], stdout=subprocess.PIPE
            )

            tleap_output, _ = tleap_process.communicate()
            tleap_exit_code = tleap_process.returncode

            with open("tleap_output.log", "w") as file:
                file.write(f"error code: {tleap_exit_code}\nstdout:\n\n")
                file.write(tleap_output.decode())

            if not os.path.isfile(prmtop_file_name) or not os.path.isfile(
                rst7_file_name
            ):
                raise RuntimeError(f"tleap failed to execute.")

            with open("leap.log", "r") as file:

                if re.search(
                    "ERROR|WARNING|Warning|duplicate|FATAL|Could|Fatal|Error",
                    file.read(),
                ):

                    raise RuntimeError(f"tleap failed to execute.")

        return (
            os.path.join(directory, prmtop_file_name),
            os.path.join(directory, rst7_file_name),
        )

    def execute(self, directory, available_resources):

        from openforcefield.topology import Molecule, Topology

        with open(self.force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, TLeapForceFieldSource):

            raise ValueError(
                "Only TLeap force field sources are supported by this protocol."
            )

        # Load in the systems coordinates / topology
        openmm_pdb_file = app.PDBFile(self.coordinate_file_path)

        # Create an OFF topology for better insight into the layout of the system topology.
        unique_molecules = [
            Molecule.from_smiles(component.smiles)
            for component in self.substance.components
        ]

        topology = Topology.from_openmm(openmm_pdb_file.topology, unique_molecules)

        # Find a unique instance of each topology molecule to get the correct
        # atom orderings.
        topology_molecules = dict()

        for topology_molecule in topology.topology_molecules:
            topology_molecules[
                topology_molecule.reference_molecule.to_smiles()
            ] = topology_molecule

        system_templates = {}

        cutoff = pint_quantity_to_openmm(force_field_source.cutoff)

        for index, (smiles, topology_molecule) in enumerate(topology_molecules.items()):

            component_directory = os.path.join(directory, str(index))

            if os.path.isdir(component_directory):
                shutil.rmtree(component_directory)

            os.makedirs(component_directory, exist_ok=True)

            if smiles != "O" and smiles != "[H]O[H]":

                initial_mol2_name = "initial.mol2"
                initial_mol2_path = os.path.join(component_directory, initial_mol2_name)

                self._topology_molecule_to_mol2(
                    topology_molecule, initial_mol2_path, self.charge_backend
                )
                prmtop_path, _ = self._run_tleap(
                    force_field_source, initial_mol2_name, component_directory
                )

                prmtop_file = openmm.app.AmberPrmtopFile(prmtop_path)

                component_system = prmtop_file.createSystem(
                    nonbondedMethod=app.PME,
                    nonbondedCutoff=cutoff,
                    constraints=app.HBonds,
                    rigidWater=True,
                    removeCMMotion=False,
                )

                if openmm_pdb_file.topology.getPeriodicBoxVectors() is not None:
                    component_system.setDefaultPeriodicBoxVectors(
                        *openmm_pdb_file.topology.getPeriodicBoxVectors()
                    )
            else:

                component_system = self._build_tip3p_system(
                    topology_molecule,
                    cutoff,
                    openmm_pdb_file.topology.getUnitCellDimensions(),
                )

            system_templates[unique_molecules[index].to_smiles()] = component_system

            with open(os.path.join(component_directory, f"component.xml"), "w") as file:
                file.write(openmm.XmlSerializer.serialize(component_system))

        # Create the full system object from the component templates.
        system = None

        for topology_molecule in topology.topology_molecules:

            system_template = system_templates[
                topology_molecule.reference_molecule.to_smiles()
            ]

            if system is None:

                # If no system has been set up yet, just use the first template.
                system = copy.deepcopy(system_template)
                continue

            # Append the component template to the full system.
            self._append_system(system, system_template)

        # Serialize the system object.
        system_xml = openmm.XmlSerializer.serialize(system)

        self.system_path = os.path.join(directory, "system.xml")

        with open(self.system_path, "w") as file:
            file.write(system_xml)

        return self._get_output_dictionary()
