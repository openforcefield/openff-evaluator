"""
A collection of protocols for assigning force field parameters to molecular
systems.
"""
import abc
import io
import logging
import os
import re
import subprocess
from enum import Enum

import numpy as np
import openmm
import requests
from openff.units import unit
from openff.units.openmm import to_openmm
from openmm import app

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import (
    ForceFieldSource,
    LigParGenForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.forcefield.forcefield import FoyerForceFieldSource
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.substances import Substance
from openff.evaluator.utils.utils import (
    get_data_filename,
    has_openeye,
    temporarily_change_directory,
)
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute

logger = logging.getLogger(__name__)


@workflow_protocol()
class BaseBuildSystem(Protocol, abc.ABC):
    """The base class for any protocol whose role is to apply a set of
    force field parameters to a given system.
    """

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

    parameterized_system = OutputAttribute(
        docstring="The parameterized system object.", type_hint=ParameterizedSystem
    )

    @staticmethod
    def _append_system(existing_system, system_to_append, cutoff, index_map=None):
        """Appends a system object onto the end of an existing system.

        Parameters
        ----------
        existing_system: openmm.System, optional
            The base system to extend.
        system_to_append: openmm.System
            The system to append.
        cutoff: openff.evaluator.unit.Quantity
            The nonbonded cutoff
        index_map: dict of int and int, optional
            A map to apply to the indices of atoms in the `system_to_append`.
            This is predominantly to be used when the ordering of the atoms
            in the `system_to_append` does not match the ordering in the full
            topology.
        """
        supported_force_types = [
            openmm.HarmonicBondForce,
            openmm.HarmonicAngleForce,
            openmm.PeriodicTorsionForce,
            openmm.NonbondedForce,
            openmm.RBTorsionForce,
            openmm.CustomNonbondedForce,
            openmm.CustomBondForce,
        ]

        number_of_appended_forces = 0
        index_offset = existing_system.getNumParticles()

        # Create an index map if one is not provided.
        if index_map is None:
            index_map = {i: i for i in range(system_to_append.getNumParticles())}

        # Append the particles.
        for index in range(system_to_append.getNumParticles()):
            index = index_map[index]
            existing_system.addParticle(system_to_append.getParticleMass(index))

        # Append the constraints
        for index in range(system_to_append.getNumConstraints()):
            index_a, index_b, distance = system_to_append.getConstraintParameters(index)

            index_a = index_map[index_a]
            index_b = index_map[index_b]

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

                if isinstance(
                    force_to_append, openmm.CustomNonbondedForce
                ) or isinstance(force_to_append, openmm.CustomBondForce):
                    if force_to_append.getEnergyFunction() != force.getEnergyFunction():
                        continue

                existing_force = force
                break

            if existing_force is None:
                if isinstance(force_to_append, openmm.CustomNonbondedForce):
                    existing_force = openmm.CustomNonbondedForce(
                        force_to_append.getEnergyFunction()
                    )
                    existing_force.setCutoffDistance(cutoff)
                    existing_force.setNonbondedMethod(
                        openmm.CustomNonbondedForce.CutoffPeriodic
                    )
                    for index in range(force_to_append.getNumGlobalParameters()):
                        existing_force.addGlobalParameter(
                            force_to_append.getGlobalParameterName(index),
                            force_to_append.getGlobalParameterDefaultValue(index),
                        )
                    for index in range(force_to_append.getNumPerParticleParameters()):
                        existing_force.addPerParticleParameter(
                            force_to_append.getPerParticleParameterName(index)
                        )
                    existing_system.addForce(existing_force)

                elif isinstance(force_to_append, openmm.CustomBondForce):
                    existing_force = openmm.CustomBondForce(
                        force_to_append.getEnergyFunction()
                    )
                    for index in range(force_to_append.getNumGlobalParameters()):
                        existing_force.addGlobalParameter(
                            force_to_append.getGlobalParameterName(index),
                            force_to_append.getGlobalParameterDefaultValue(index),
                        )
                    for index in range(force_to_append.getNumPerBondParameters()):
                        existing_force.addPerBondParameter(
                            force_to_append.getPerBondParameterName(index)
                        )
                    existing_system.addForce(existing_force)

                else:
                    existing_force = type(force_to_append)()
                    existing_system.addForce(existing_force)

            if isinstance(force_to_append, openmm.HarmonicBondForce):
                # Add the bonds.
                for index in range(force_to_append.getNumBonds()):
                    index_a, index_b, *parameters = force_to_append.getBondParameters(
                        index
                    )

                    index_a = index_map[index_a]
                    index_b = index_map[index_b]

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

                    index_a = index_map[index_a]
                    index_b = index_map[index_b]
                    index_c = index_map[index_c]

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

                    index_a = index_map[index_a]
                    index_b = index_map[index_b]
                    index_c = index_map[index_c]
                    index_d = index_map[index_d]

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
                    index = index_map[index]

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

                    index_a = index_map[index_a]
                    index_b = index_map[index_b]

                    existing_force.addException(
                        index_a + index_offset, index_b + index_offset, *parameters
                    )

            elif isinstance(force_to_append, openmm.RBTorsionForce):
                # Support for RBTorisionForce needed for OPLSAA, etc
                for index in range(force_to_append.getNumTorsions()):
                    torsion_params = force_to_append.getTorsionParameters(index)
                    for i in range(4):
                        torsion_params[i] = index_map[torsion_params[i]] + index_offset

                    existing_force.addTorsion(*torsion_params)

            elif isinstance(force_to_append, openmm.CustomNonbondedForce):
                for index in range(force_to_append.getNumParticles()):
                    nb_params = force_to_append.getParticleParameters(index_map[index])
                    existing_force.addParticle(nb_params)

                # Add the 1-2, 1-3 and 1-4 exceptions.
                for index in range(force_to_append.getNumExclusions()):
                    (
                        index_a,
                        index_b,
                    ) = force_to_append.getExclusionParticles(index)

                    index_a = index_map[index_a]
                    index_b = index_map[index_b]

                    existing_force.addExclusion(
                        index_a + index_offset, index_b + index_offset
                    )

            elif isinstance(force_to_append, openmm.CustomBondForce):
                for index in range(force_to_append.getNumBonds()):
                    index_a, index_b, bond_params = force_to_append.getBondParameters(
                        index
                    )

                    index_a = index_map[index_a] + index_offset
                    index_b = index_map[index_b] + index_offset

                    existing_force.addBond(index_a, index_b, bond_params)

            number_of_appended_forces += 1

        if number_of_appended_forces != system_to_append.getNumForces():
            raise ValueError("Not all forces were appended.")

    def _execute(self, directory, available_resources):
        raise NotImplementedError()


@workflow_protocol()
class TemplateBuildSystem(BaseBuildSystem, abc.ABC):
    """A base protocol for any protocol which assign parameters to a system
    by first assigning parameters to each individual component of the system,
    and then replicating those templates for each instance of the component.
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

    water_model = InputAttribute(
        docstring="The water model to apply, if any water molecules are present.",
        type_hint=WaterModel,
        default_value=WaterModel.TIP3P,
    )

    @staticmethod
    def _build_tip3p_system(cutoff, cell_vectors):
        """Builds a `openmm.System` object containing a single water model

        Parameters
        ----------
        cutoff: openff.evaluator.unit.Quantity
            The non-bonded cutoff.
        cell_vectors: openff.evaluator.unit.Quantity
            The full system's cell vectors.

        Returns
        -------
        openmm.System
            The created system.
        """

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
    def _create_empty_system(cutoff):
        """Creates an empty system object with stub forces.

        Parameters
        ----------
        cutoff: openmm.unit
            The non-bonded cutoff.

        Returns
        -------
        openmm.System
            The created system object.
        """

        system = openmm.System()

        system.addForce(openmm.HarmonicBondForce())
        system.addForce(openmm.HarmonicAngleForce())
        system.addForce(openmm.PeriodicTorsionForce())

        nonbonded_force = openmm.NonbondedForce()
        nonbonded_force.setCutoffDistance(cutoff)
        nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)

        system.addForce(nonbonded_force)

        return system

    @abc.abstractmethod
    def _parameterize_molecule(self, molecule, force_field_source, cutoff):
        """Parameterize the specified molecule.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to parameterize.
        force_field_source: ForceFieldSource
            The tleap source which describes which parameters to apply.
        cutoff: openmm.unit
            The non-bonded cutoff.

        Returns
        -------
        openmm.System
            The parameterized system.
        """
        raise NotImplementedError()

    def _execute(self, directory, available_resources):
        from openff.toolkit.topology import Molecule, Topology
        from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper

        force_field_source = ForceFieldSource.from_json(self.force_field_path)
        cutoff = to_openmm(force_field_source.cutoff)

        # Load in the systems topology
        openmm_pdb_file = app.PDBFile(self.coordinate_file_path)

        # Create an OFF topology for better insight into the layout of the system
        # topology.
        unique_molecules = {}

        for component in self.substance:
            unique_molecule = Molecule.from_smiles(
                component.smiles, toolkit_registry=RDKitToolkitWrapper()
            )
            unique_molecules[
                unique_molecule.to_smiles(toolkit_registry=RDKitToolkitWrapper())
            ] = unique_molecule

        # Parameterize each component in the system.
        system_templates = {}

        for index, (smiles, unique_molecule) in enumerate(unique_molecules.items()):
            if smiles in ["O", "[H]O[H]", "[H][O][H]"]:
                component_system = self._build_tip3p_system(
                    cutoff,
                    openmm_pdb_file.topology.getUnitCellDimensions(),
                )

            else:
                component_directory = os.path.join(directory, str(index))
                os.makedirs(component_directory, exist_ok=True)

                with temporarily_change_directory(component_directory):
                    component_system = self._parameterize_molecule(
                        unique_molecule, force_field_source, cutoff
                    )

            system_templates[smiles] = component_system

        # Apply the parameters to the topology.
        topology = Topology.from_openmm(
            openmm_pdb_file.topology, unique_molecules.values()
        )

        # Create the full system object from the component templates.
        system = self._create_empty_system(cutoff)

        groupings = topology.identical_molecule_groups

        for unique_molecule_index, group in groupings.items():
            unique_molecule = topology.molecule(unique_molecule_index)

            smiles = unique_molecule.to_smiles(toolkit_registry=RDKitToolkitWrapper())
            system_template = system_templates[smiles]

            index_map = {}
            # {0: [[0, {0: 0, 1: 1, 2: 2}],
            #     [1, {0: 0, 1: 1, 2: 2}],
            #     [2, {0: 0, 1: 1, 2: 2}]]}

            for duplicate_molecule_index, _ in group:
                duplicate_molecule = topology.molecule(duplicate_molecule_index)

                for index, atom in enumerate(duplicate_molecule.atoms):
                    index_map[atom.molecule_particle_index] = index

                self._append_system(system, system_template, cutoff, index_map)

        if openmm_pdb_file.topology.getPeriodicBoxVectors() is not None:
            system.setDefaultPeriodicBoxVectors(
                *openmm_pdb_file.topology.getPeriodicBoxVectors()
            )

        # Serialize the system object.
        system_path = os.path.join(directory, "system.xml")

        with open(system_path, "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))

        self.parameterized_system = ParameterizedSystem(
            substance=self.substance,
            force_field=force_field_source,
            topology_path=self.coordinate_file_path,
            system_path=system_path,
        )


@workflow_protocol()
class BuildSmirnoffSystem(BaseBuildSystem):
    """Parametrise a set of molecules with a given smirnoff force field
    using the `OpenFF toolkit <https://github.com/openforcefield/openff-toolkit>`_.
    """

    def _execute(self, directory, available_resources):
        from openff.toolkit.topology import Molecule, Topology

        pdb_file = app.PDBFile(self.coordinate_file_path)

        force_field_source = ForceFieldSource.from_json(self.force_field_path)

        if not isinstance(force_field_source, SmirnoffForceFieldSource):
            raise ValueError(
                "Only SMIRNOFF force fields are supported by this protocol."
            )

        force_field = force_field_source.to_force_field()

        # Create the molecules to parameterize from the input substance.
        unique_molecules = []

        for component in self.substance.components:
            molecule = Molecule.from_smiles(smiles=component.smiles)

            if molecule is None:
                raise ValueError(f"{component} could not be converted to a Molecule")

            unique_molecules.append(molecule)

        # Create the topology to parameterize from the input coordinates and the
        # expected molecule species.
        topology = Topology.from_openmm(
            pdb_file.topology, unique_molecules=unique_molecules
        )

        system = force_field.create_openmm_system(topology)

        if system is None:
            raise RuntimeError(
                "Failed to create a system from the specified topology and molecules."
            )

        system_xml = openmm.XmlSerializer.serialize(system)
        system_path = os.path.join(directory, "system.xml")

        with open(system_path, "w") as file:
            file.write(system_xml)

        self.parameterized_system = ParameterizedSystem(
            substance=self.substance,
            force_field=force_field_source,
            topology_path=self.coordinate_file_path,
            system_path=system_path,
        )


@workflow_protocol()
class BuildLigParGenSystem(TemplateBuildSystem):
    """Parametrise a set of molecules with the OPLS-AA/M force field.
    using a `LigParGen server <http://zarbi.chem.yale.edu/ligpargen/>`_.

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
    def _built_template(molecule, force_field_source):
        """Builds a force field template object.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to templatize.
        force_field_source: LigParGenForceFieldSource
            The tleap source which describes which parameters to apply.

        Returns
        -------
        openmm.app.ForceField
            The force field template.
        """
        initial_request_url = force_field_source.request_url
        empty_stream = io.BytesIO(b"\r\n")

        total_charge = molecule.total_charge

        if isinstance(total_charge, unit.Quantity):
            total_charge = total_charge.m_as(unit.elementary_charge)

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
                logger.warning(
                    f"The preferred charge model is {str(force_field_source.preferred_charge_model)}, "
                    f"however the system is charged and so the "
                    f"{str(LigParGenForceFieldSource.ChargeModel.CM1A_1_14)} model will be used in its "
                    f"place."
                )

        data_body = {
            "smiData": (None, molecule.to_smiles()),
            "molpdbfile": ("", empty_stream),
            "checkopt": (None, 0),
            "chargetype": (None, charge_model),
            "dropcharge": (None, str(total_charge)),
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
        force_field_path = "template.xml"

        with open(force_field_path, "wb") as file:
            file.write(force_field_response)

        return app.ForceField(force_field_path)

    def _parameterize_molecule(self, molecule, force_field_source, cutoff):
        """Parameterize the specified molecule.

        Parameters
        ----------
        force_field_source: LigParGenForceFieldSource
            The tleap source which describes which parameters to apply.

        Returns
        -------
        openmm.System
            The parameterized system.
        """
        from openmm import unit as openmm_unit

        template = self._built_template(molecule, force_field_source)

        off_topology = molecule.to_topology()

        box_vectors = np.eye(3) * 10.0

        openmm_topology = off_topology.to_openmm()
        openmm_topology.setPeriodicBoxVectors(box_vectors * openmm_unit.nanometer)

        system = template.createSystem(
            topology=openmm_topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
        )

        with open("component.xml", "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))

        return system

    @staticmethod
    def _apply_opls_mixing_rules(system):
        """Applies the OPLS mixing rules to the system.

        Notes
        -----
        This method is based upon that found in the `LigParGen tutorial
        <http://zarbi.chem.yale.edu/ligpargen/openMM_tutorial.html>`_.

        Parameters
        ----------
        system: openmm.System
            The system object to apply the OPLS mixing rules to.
        """
        from openmm import unit as openmm_unit

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
                    epsilon.value_in_unit(openmm_unit.kilojoule_per_mole), 0.0
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

    def _execute(self, directory, available_resources):
        with open(self.force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, LigParGenForceFieldSource):
            raise ValueError(
                "Only LigParGen force field sources are supported by this protocol."
            )

        super(BuildLigParGenSystem, self)._execute(directory, available_resources)

        with open(self.parameterized_system.system_path) as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        # Apply the OPLS mixing rules.
        self._apply_opls_mixing_rules(system)

        with open(self.parameterized_system.system_path, "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))


@workflow_protocol()
class BuildTLeapSystem(TemplateBuildSystem):
    """Parametrise a set of molecules with an Amber based force field.
    using the `tleap package <http://ambermd.org/AmberTools.php>`_.

    Notes
    -----
    * This protocol is currently a work in progress and as such has limited
      functionality compared to the more established `BuildSmirnoffSystem` protocol.
    * This protocol requires the optional `ambertools >=19.0` dependency to be installed.
    """

    class ChargeBackend(Enum):
        """The framework to use to assign partial charges."""

        OpenEye = "OpenEye"
        AmberTools = "AmberTools"

    charge_backend = InputAttribute(
        docstring="The backend framework to use to assign partial charges.",
        type_hint=ChargeBackend,
        default_value=lambda: BuildTLeapSystem.ChargeBackend.OpenEye
        if has_openeye()
        else BuildTLeapSystem.ChargeBackend.AmberTools,
    )

    @staticmethod
    def _run_tleap(molecule, force_field_source, directory):
        """Uses tleap to apply parameters to a particular molecule,
        generating a `.prmtop` and a `.rst7` file with the applied parameters.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to parameterize.
        force_field_source: TLeapForceFieldSource
            The tleap source which describes which parameters to apply.
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
            initial_file_path = "initial.sdf"
            molecule.to_file(initial_file_path, file_format="SDF")

            # Save the molecule charges to a file.
            charges = [x.m_as(unit.elementary_charge) for x in molecule.partial_charges]

            with open("charges.txt", "w") as file:
                file.write("\n".join(map(str, charges)))
                file.write("\n")

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
                    initial_file_path,
                    "-fi",
                    "sdf",
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
                    "-c",
                    "rc",
                    "-cf",
                    "charges.txt",
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
                    f"({initial_file_path})"
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
                template_lines.append(
                    f"loadamberparams {frcmod_path}",
                )

            prmtop_file_name = "structure.prmtop"
            rst7_file_name = "structure.rst7"

            template_lines.extend(
                [
                    f"MOL = loadmol2 {processed_mol2_path}",
                    'setBox MOL "centers"',
                    "check MOL",
                    f"saveamberparm MOL {prmtop_file_name} {rst7_file_name}",
                ]
            )

            input_file_path = "tleap.in"

            with open(input_file_path, "w") as file:
                file.write("\n".join(template_lines))
                file.write("\nquit\n")

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
                raise RuntimeError("tleap failed to execute.")

            with open("leap.log", "r") as file:
                if re.search(
                    "Exiting LEaP: Errors = 0; Warnings = 0; Notes = 0.", file.read()
                ):
                    # normal exiting log
                    pass
                elif re.search(
                    "ERROR|WARNING|Warning|duplicate|FATAL|Could|Fatal|Error",
                    file.read(),
                ):
                    raise RuntimeError("tleap failed to execute.")

        return (
            os.path.join(directory, prmtop_file_name),
            os.path.join(directory, rst7_file_name),
        )

    def _generate_charges(self, molecule):
        """Generates a set of partial charges for a molecule using
        the specified charge backend.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to assign charges to.
        """

        if self.charge_backend == BuildTLeapSystem.ChargeBackend.OpenEye:
            from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper

            toolkit_wrapper = OpenEyeToolkitWrapper()

        elif self.charge_backend == BuildTLeapSystem.ChargeBackend.AmberTools:
            from openff.toolkit.utils.toolkits import (
                AmberToolsToolkitWrapper,
                RDKitToolkitWrapper,
                ToolkitRegistry,
            )

            toolkit_wrapper = ToolkitRegistry(
                toolkit_precedence=[RDKitToolkitWrapper, AmberToolsToolkitWrapper]
            )

        else:
            raise ValueError("Invalid toolkit specification.")

        molecule.generate_conformers(toolkit_registry=toolkit_wrapper)
        molecule.assign_partial_charges(
            partial_charge_method="am1bcc",
            toolkit_registry=toolkit_wrapper,
            normalize_partial_charges=True,
        )

    def _parameterize_molecule(self, molecule, force_field_source, cutoff):
        """Parameterize the specified molecule.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to parameterize.
        force_field_source: TLeapForceFieldSource
            The tleap source which describes which parameters to apply.

        Returns
        -------
        openmm.System
            The parameterized system.
        """

        self._generate_charges(molecule)

        prmtop_path, _ = BuildTLeapSystem._run_tleap(molecule, force_field_source, "")
        prmtop_file = openmm.app.AmberPrmtopFile(prmtop_path)

        system = prmtop_file.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
        )

        with open("component.xml", "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))

        return system

    def _execute(self, directory, available_resources):
        force_field_source = ForceFieldSource.from_json(self.force_field_path)

        if not isinstance(force_field_source, TLeapForceFieldSource):
            raise ValueError(
                "Only TLeap force field sources are supported by this protocol."
            )

        super(BuildTLeapSystem, self)._execute(directory, available_resources)


@workflow_protocol()
class BuildFoyerSystem(TemplateBuildSystem):
    """Parameterize a set of molecules with a Foyer force field source"""

    def _parameterize_molecule(self, molecule, force_field_source, cutoff):
        """Parameterize the specified molecule.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to parameterize.
        force_field_source: FoyerForceFieldSource
            The foyer source which describes which parameters to apply.

        Returns
        -------
        openmm.System
            The parameterized system.
        """
        from foyer import Forcefield as FoyerForceField
        from openff.interchange import Interchange
        from openff.toolkit import Topology

        topology: Topology = molecule.to_topology()

        force_field: FoyerForceField
        if force_field_source.foyer_source.lower() == "oplsaa":
            force_field = FoyerForceField(name="oplsaa")
        else:
            force_field = FoyerForceField(
                forcefield_files=force_field_source.foyer_source
            )

        interchange = Interchange.from_foyer(topology=topology, force_field=force_field)

        interchange.box = [10, 10, 10] * unit.nanometers

        openmm_system = interchange.to_openmm(combine_nonbonded_forces=False)

        return openmm_system

    def _execute(self, directory, available_resources):
        force_field_source = ForceFieldSource.from_json(self.force_field_path)

        if not isinstance(force_field_source, FoyerForceFieldSource):
            raise ValueError(
                "Only Foyer force field sources are supported by this protocol."
            )

        super(BuildFoyerSystem, self)._execute(directory, available_resources)
