"""
A set of utilities for helping to perform simulations using openmm.
"""
import copy
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy
import openmm
from openff.units import unit
from openff.units.openmm import from_openmm
from openmm import app
from openmm import unit as openmm_unit

from openff.evaluator.attributes.attributes import UndefinedAttribute
from openff.evaluator.forcefield import ParameterGradientKey

if TYPE_CHECKING:
    from openff.toolkit.topology import Topology
    from openff.toolkit.typing.engines.smirnoff import ForceField

logger = logging.getLogger(__name__)


def setup_platform_with_resources(compute_resources, high_precision=False):
    """Creates an OpenMM `Platform` object which requests a set
    amount of compute resources (e.g with a certain number of cpus).

    Parameters
    ----------
    compute_resources: ComputeResources
        The compute resources which describe which platform is most
        appropriate.
    high_precision: bool
        If true, a platform with the highest possible precision (double
        for CUDA and OpenCL, Reference for CPU only) will be returned.
        For GPU platforms, this overrides the precision level in
        `compute_resources`.

    Returns
    -------
    Platform
        The created platform
    """
    from openmm import Platform

    # Setup the requested platform:
    if compute_resources.number_of_gpus > 0:
        # TODO: Deterministic forces = True

        from openff.evaluator.backends import ComputeResources

        toolkit_enum = ComputeResources.GPUToolkit(
            compute_resources.preferred_gpu_toolkit
        )
        precision_level = ComputeResources.GPUPrecision(
            compute_resources.preferred_gpu_precision
        )

        # Get platform for running on GPUs.
        if toolkit_enum == ComputeResources.GPUToolkit.auto:
            from openmmtools.utils import get_fastest_platform

            # noinspection PyCallByClass,PyTypeChecker
            platform = get_fastest_platform(minimum_precision=precision_level)

        elif toolkit_enum == ComputeResources.GPUToolkit.CUDA:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName("CUDA")

        elif toolkit_enum == ComputeResources.GPUToolkit.OpenCL:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName("OpenCL")

        else:
            raise KeyError(f"Specified GPU toolkit {toolkit_enum} is not supported.")

        # Set GPU device index
        if compute_resources.gpu_device_indices is not None:
            # `DeviceIndex` is used by both CUDA and OpenCL
            platform.setPropertyDefaultValue(
                "DeviceIndex",
                compute_resources.gpu_device_indices,
            )

        # Set GPU precision level
        platform.setPropertyDefaultValue("Precision", precision_level.value)
        if high_precision:
            platform.setPropertyDefaultValue("Precision", "double")

        # Print platform information
        logger.info(
            "Setting up an openmm platform on GPU {} with {} kernel and {} precision".format(
                compute_resources.gpu_device_indices or 0,
                platform.getName(),
                platform.getPropertyDefaultValue("Precision"),
            )
        )

    else:
        if not high_precision:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName("CPU")
            platform.setPropertyDefaultValue(
                "Threads", str(compute_resources.number_of_threads)
            )

        else:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName("Reference")

        logger.info(
            "Setting up a simulation with {} threads".format(
                compute_resources.number_of_threads
            )
        )

    return platform


def disable_pbc(system):
    """Disables any periodic boundary conditions being applied
    to non-bonded forces by setting the non-bonded method to
    `NoCutoff = 0`

    Parameters
    ----------
    system: openmm.system
        The system which should have periodic boundary conditions
        disabled.
    """

    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)

        if (
            not isinstance(force, openmm.NonbondedForce)
            and not isinstance(force, openmm.CustomGBForce)
            and not isinstance(force, openmm.GBSAOBCForce)
        ):
            continue

        force.setNonbondedMethod(
            0
        )  # NoCutoff = 0, NonbondedMethod.CutoffNonPeriodic = 1


def system_subset(
    parameter_key: ParameterGradientKey,
    force_field: "ForceField",
    topology: "Topology",
    scale_amount: Optional[float] = None,
) -> Tuple["openmm.System", "unit.Quantity"]:
    """Produces an OpenMM system containing the minimum number of forces while
    still containing a specified force field parameter, and those other parameters
    which may interact with it (e.g. in the case of vdW parameters).

    The value of the parameter of interest may optionally be perturbed by an amount
    specified by ``scale_amount``.

    Parameters
    ----------
    parameter_key
        The parameter of interest.
    force_field
        The force field to create the system from (and optionally perturb).
    topology
        The topology of the system to apply the force field to.
    scale_amount: float, optional
        The optional amount to perturb the ``parameter`` by such that
        ``parameter = (1.0 + scale_amount) * parameter``.

    Returns
    -------
        The created system as well as the value of the specified ``parameter``.
    """

    # As this method deals mainly with the toolkit, we stick to
    # openmm units here.
    from openff.toolkit.typing.engines.smirnoff import ForceField

    # Create the force field subset.
    force_field_subset = ForceField(load_plugins=True)

    handlers_to_register = {parameter_key.tag}

    if parameter_key.tag in {"ChargeIncrementModel", "LibraryCharges"}:
        # Make sure to retain all of the electrostatic handlers when dealing with
        # charges as the applied charges will depend on which charges have been applied
        # by previous handlers.
        handlers_to_register.update(
            {"Electrostatics", "ChargeIncrementModel", "LibraryCharges"}
        )

    if parameter_key.tag in {"GBSA", "CustomGBSA"}:
        # CustomGBForce and GBSAOBCForce requires the nonbonded parameters
        # loaded otherwise the parameters will be set to zero.
        handlers_to_register.update(
            {
                "vdW",
                "Electrostatics",
                "ChargeIncrementModel",
                "LibraryCharges",
                "ToolkitAM1BCC",
            }
        )

    registered_handlers = force_field.registered_parameter_handlers

    for handler_to_register in handlers_to_register:
        if handler_to_register not in registered_handlers:
            continue

        force_field_subset.register_parameter_handler(
            copy.deepcopy(force_field.get_parameter_handler(handler_to_register))
        )

    handler = force_field_subset.get_parameter_handler(parameter_key.tag)

    if handler._OPENMMTYPE == openmm.CustomNonbondedForce:
        vdw_handler = force_field_subset.get_parameter_handler("vdW")
        # we need a generic blank parameter to work around this toolkit issue
        # <https://github.com/openforcefield/openff-toolkit/issues/1102>
        vdw_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[*:1]",
                "epsilon": 0.0 * unit.kilocalories_per_mole,
                "sigma": 1.0 * unit.angstrom,
            }
        )

    parameter = (
        handler
        if parameter_key.smirks is None
        else handler.parameters[parameter_key.smirks]
    )

    parameter_value = getattr(parameter, parameter_key.attribute)
    is_quantity = isinstance(parameter_value, openmm_unit.Quantity)

    if not is_quantity:
        parameter_value = parameter_value * openmm_unit.dimensionless

    # Optionally perturb the parameter of interest.
    if scale_amount is not None:
        if numpy.isclose(parameter_value.value_in_unit(parameter_value.unit), 0.0):
            # Careful thought needs to be given to this. Consider cases such as
            # epsilon or sigma where negative values are not allowed.
            parameter_value = (
                scale_amount if scale_amount > 0.0 else 0.0
            ) * parameter_value.unit
        else:
            parameter_value *= 1.0 + scale_amount

    # Pretty sure Pint doesn't do this, but need to check
    if not isinstance(parameter_value, openmm_unit.Quantity):
        # Handle the case where OMM down-converts a dimensionless quantity to a float.
        parameter_value = parameter_value * openmm_unit.dimensionless

    setattr(
        parameter,
        parameter_key.attribute,
        parameter_value
        if is_quantity
        else parameter_value.value_in_unit(openmm_unit.dimensionless),
    )
    logger.info("The parameter value is {parameter_value}")

    # Create the parameterized sub-system.
    system = force_field_subset.create_openmm_system(topology)
    return system, parameter_value


def update_context_with_positions(
    context: openmm.Context,
    positions: openmm_unit.Quantity,
    box_vectors: Optional[openmm_unit.Quantity],
):
    """Set a collection of positions and box vectors on an OpenMM context and compute
    any extra positions such as v-site positions.
    Parameters
    ----------
    context
        The OpenMM context to set the positions on.
    positions
        A unit wrapped numpy array with shape=(n_atoms, 3) that contains the positions
        to set.
    box_vectors
        An optional unit wrapped numpy array with shape=(3, 3) that contains the box
        vectors to set.
    """

    system = context.getSystem()

    n_vsites = sum(
        1 for i in range(system.getNumParticles()) if system.isVirtualSite(i)
    )

    n_atoms = system.getNumParticles() - n_vsites

    if len(positions) != n_atoms and len(positions) != (n_atoms + n_vsites):
        raise ValueError(
            "The length of the positions array does not match either the "
            "the number of atoms or the number of atoms + v-sites."
        )

    if n_vsites > 0 and len(positions) != (n_atoms + n_vsites):
        new_positions = numpy.zeros((system.getNumParticles(), 3))

        i = 0

        for j in range(system.getNumParticles()):
            if not system.isVirtualSite(j):
                # take an old position and update the index
                new_positions[j] = positions[i].value_in_unit(openmm_unit.nanometers)
                i += 1

        positions = new_positions * openmm_unit.nanometers

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)

    context.setPositions(positions)

    if n_vsites > 0:
        context.computeVirtualSites()


def update_context_with_pdb(
    context: openmm.Context,
    pdb_file: app.PDBFile,
):
    """Extracts the positions and box vectors from a PDB file object and set these
    on an OpenMM context and compute any extra positions such as v-site positions.
    Parameters
    ----------
    context
        The OpenMM context to set the positions on.
    pdb_file
        The PDB file object to extract the positions and box vectors from.
    """

    positions = pdb_file.getPositions(asNumpy=True)

    box_vectors = pdb_file.topology.getPeriodicBoxVectors()

    if box_vectors is None:
        box_vectors = context.getSystem().getDefaultPeriodicBoxVectors()

    update_context_with_positions(context, positions, box_vectors)


def extract_atom_indices(system: openmm.System) -> List[int]:
    """Returns the indices of atoms in a system excluding any virtual sites."""

    return [i for i in range(system.getNumParticles()) if not system.isVirtualSite(i)]


def extract_positions(
    state: openmm.State,
    particle_indices: Optional[List[int]] = None,
) -> openmm_unit.Quantity:
    """Extracts the positions from an OpenMM context, optionally excluding any v-site
    positions which should be uniquely defined by the atomic positions.
    """

    positions = state.getPositions(asNumpy=True)

    if particle_indices is not None:
        positions = positions[particle_indices]

    return positions


def openmm_quantity_to_pint(openmm_quantity):
    """Converts a `openmm.unit.Quantity` to a `openff.evaluator.unit.Quantity`.
    Parameters
    ----------
    openmm_quantity: openmm.unit.Quantity
        The quantity to convert.
    Returns
    -------
    openff.evaluator.unit.Quantity
        The converted quantity.
    """

    # from openff.units.openmm import from_openmm

    if openmm_quantity is None or isinstance(openmm_quantity, UndefinedAttribute):
        return None

    return from_openmm(openmm_quantity)


def pint_quantity_to_openmm(pint_quantity):
    """Converts a `openff.evaluator.unit.Quantity` to a `openmm.unit.Quantity`.
    Notes
    -----
    Not all pint units are available in OpenMM.
    Parameters
    ----------
    pint_quantity: openff.evaluator.unit.Quantity
        The quantity to convert.
    Returns
    -------
    openmm.unit.Quantity
        The converted quantity.
    """

    from openff.units.openmm import to_openmm

    if pint_quantity is None or isinstance(pint_quantity, UndefinedAttribute):
        return None

    return to_openmm(pint_quantity)


def openmm_unit_to_string(input_unit: openmm.unit.Unit) -> str:
    """
    Serialize an openmm.unit.Unit to string.
    Taken from https://github.com/openforcefield/openff-toolkit/blob/97462b88a4b50a608e10f735ee503c655f9b64d3/openff/toolkit/utils/utils.py#L170-L204
    """
    if input_unit == openmm_unit.dimensionless:
        return "dimensionless"

    # Decompose output_unit into a tuples of (base_dimension_unit, exponent)
    unit_string = None

    for unit_component in input_unit.iter_base_or_scaled_units():
        unit_component_name = unit_component[0].name
        # Convert, for example "elementary charge" --> "elementary_charge"
        unit_component_name = unit_component_name.replace(" ", "_")
        if unit_component[1] == 1:
            contribution = "{}".format(unit_component_name)
        else:
            contribution = "{}**{}".format(unit_component_name, int(unit_component[1]))
        if unit_string is None:
            unit_string = contribution
        else:
            unit_string += " * {}".format(contribution)

    return unit_string


def openmm_quantity_to_string(input_quantity: openmm.unit.Quantity) -> str:
    """
    Serialize a openmm.unit.Quantity to string.
    Taken from https://github.com/openforcefield/openff-toolkit/blob/97462b88a4b50a608e10f735ee503c655f9b64d3/openff/toolkit/utils/utils.py
    """
    if input_quantity is None:
        return None

    unitless_value = input_quantity.value_in_unit(input_quantity.unit)

    if isinstance(unitless_value, numpy.ndarray):
        unitless_value = list(unitless_value)

    unit_string = openmm_unit_to_string(input_quantity.unit)

    return f"{unitless_value} * {unit_string}"


def perturbed_gaff_system(
    parameter_key: ParameterGradientKey,
    system_path: str,
    topology_path: str,
    enable_pbc: bool,
    scale_amount: Optional[float] = None,
) -> Tuple["openmm.System", "openmm_unit.Quantity"]:
    """Produces an OpenMM system with perturbed parameters used in gradient
    calculations.

    The value of the parameter of interest may optionally be perturbed by an amount
    specified by ``scale_amount``.

    Parameters
    ----------
    parameter_key
        The parameter of interest.
    system_path
        The path to the OpenMM XML system.
    topology_path
        The path to the Amber topology file.
    enable_pbc
        Whether PBC should be enabled.
    scale_amount: float, optional
        The optional amount to perturb the ``parameter`` by such that
        ``parameter = (1.0 + scale_amount) * parameter``.

    Returns
    -------
        The created system as well as the value of the specified ``parameter``.
    """

    import parmed

    if not os.path.isfile(topology_path):
        logger.info(
            "GAFF topology file not found, returning an empty `system` object and "
            "setting `parameter_value` to zero."
        )

        perturbed_system = openmm.System()
        parameter_value = 0.0

        if scale_amount is not None:
            parameter_value = scale_amount if scale_amount > 0.0 else 0.0

        if parameter_key.tag == "Bond":
            if parameter_key.attribute == "length":
                parameter_value *= openmm_unit.angstrom
            elif parameter_key.attribute == "k":
                parameter_value *= (
                    openmm_unit.kilocalorie_per_mole / openmm_unit.angstrom**2
                )

        elif parameter_key.tag == "Angle":
            if parameter_key.attribute == "theta":
                parameter_value *= openmm_unit.degree
            elif parameter_key.attribute == "k":
                parameter_value *= (
                    openmm_unit.kilocalorie_per_mole / openmm_unit.radians**2
                )

        elif parameter_key.tag == "vdW":
            if parameter_key.attribute == "rmin_half":
                parameter_value *= openmm_unit.angstrom
            elif parameter_key.attribute == "epsilon":
                parameter_value *= openmm_unit.kilocalorie_per_mole

        elif parameter_key.tag == "GBSA":
            if parameter_key.attribute == "radius":
                parameter_value *= openmm_unit.nanometer
            elif parameter_key.attribute == "scale":
                parameter_value *= openmm_unit.dimensionless

        return perturbed_system, parameter_value

    # Load Topology and System XML
    structure = parmed.load_file(topology_path)
    with open(system_path, "r") as f:
        original_system = openmm.XmlSerializer.deserialize(f.read())
    perturbed_system = copy.deepcopy(original_system)

    if parameter_key.tag == "Bond":
        # Bond atom types
        bond_type = parameter_key.smirks.replace("-", " ").split()
        bond_list = []

        # Get Bond parameters
        bonds = str(parmed.tools.printBonds(structure)).split("\n")
        for i, bond in enumerate(bonds):
            if i == 0 or i == len(bonds) - 1:
                continue
            split_line = bond.split()
            atom_type1 = split_line[3].split(")")[0]
            atom_type2 = split_line[7].split(")")[0]
            if [atom_type1, atom_type2] == bond_type or [
                atom_type2,
                atom_type1,
            ] == bond_type:
                atom_index1 = int(split_line[0]) - 1
                atom_index2 = int(split_line[4]) - 1
                bond_list.append([atom_index1, atom_index2])
                kbond = 2 * float(split_line[-2])
                rbond = float(split_line[-1])

        # Set parameter_value with simtk units
        if parameter_key.attribute == "length":
            rbond *= (1.0 + scale_amount) * openmm_unit.angstrom
            parameter_value = rbond
        elif parameter_key.attribute == "k":
            kbond *= (
                (1.0 + scale_amount)
                * openmm_unit.kilocalorie_per_mole
                / openmm_unit.angstrom**2
            )
            parameter_value = kbond

        # Get bond force from system
        bond_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                bond_force

        # Assign perturbed parameter to system
        for bond_index in bond_force.getNumBonds():
            system_bond_parm = bond_force.getBondParameters(bond_index)
            atom1 = system_bond_parm[0]
            atom2 = system_bond_parm[1]
            for bond in bond_list:
                if [atom1, atom2] == bond or [atom2, atom1] == bond:
                    bond_force.setBondParameters(bond_index, atom1, atom2, rbond, kbond)

    elif parameter_key.tag == "Angle":
        # Angle atom types
        angle_type = parameter_key.smirks.replace("-", " ").split()
        angle_list = []

        # Get Angle parameters
        angles = str(parmed.tools.printAngles(structure)).split("\n")
        for i, angle in enumerate(angles):
            if i == 0 or i == len(angles) - 1:
                continue

            split_line = angle.split()
            atom_type1 = split_line[3].split(")")[0]
            atom_type2 = split_line[7].split(")")[0]
            atom_type3 = split_line[1].split(")")[0]
            if [atom_type1, atom_type2, atom_type3] == angle_type or [
                atom_type3,
                atom_type2,
                atom_type1,
            ] == angle_type:
                atom_index1 = int(split_line[0]) - 1
                atom_index2 = int(split_line[4]) - 1
                atom_index3 = int(split_line[8]) - 1
                angle_list.append([atom_index1, atom_index2, atom_index3])
                kangle = 2 * float(split_line[-2])
                theta = float(split_line[-1])

        # Set parameter_value with simtk units
        if parameter_key.attribute == "theta":
            theta *= (1.0 + scale_amount) * openmm_unit.degree
            parameter_value = theta
        elif parameter_key.attribute == "k":
            kangle *= (
                (1.0 + scale_amount)
                * openmm_unit.kilocalorie_per_mole
                / openmm_unit.radians**2
            )
            parameter_value = kangle

        # Get angle force from system
        angle_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, openmm.HarmonicAngleForce):
                angle_force = force

        # Assign perturbed parameter to system
        for angle_index in angle_force.getNumAngles():
            system_angle_parm = angle_force.getAngleParameters(angle_index)
            atom1 = system_angle_parm[0]
            atom2 = system_angle_parm[1]
            atom3 = system_angle_parm[2]
            for angle in angle_list:
                if [atom1, atom2, atom3] == angle or [atom3, atom2, atom1] == angle:
                    angle_force.setAngleParameters(
                        angle_index, atom1, atom2, atom3, theta, kangle
                    )

    elif parameter_key.tag == "Dihedral":
        raise NotImplementedError(
            f"Gradient calculations for `{parameter_key.tag}` is currently not supported with GAFF calculations."
        )

    elif parameter_key.tag == "Improper":
        raise NotImplementedError(
            f"Gradient calculations for `{parameter_key.tag}` is currently not supported with GAFF calculations."
        )

    elif parameter_key.tag == "Electrostatic":
        raise NotImplementedError(
            f"Gradient calculations for `{parameter_key.tag}` is currently not supported with GAFF calculations."
        )

    elif parameter_key.tag == "vdW":
        # Get current LJ parameters
        lj_index = structure.LJ_types[parameter_key.smirks] - 1
        lj_radius = structure.LJ_radius[lj_index]
        lj_depth = structure.LJ_depth[lj_index]

        # Determine which parameter to perturb
        if parameter_key.attribute == "rmin_half":
            lj_radius *= 1.0 + scale_amount
            parameter_value = lj_radius * openmm_unit.angstrom
        elif parameter_key.attribute == "epsilon":
            lj_depth *= 1.0 + scale_amount
            parameter_value = lj_depth * openmm_unit.kilocalorie_per_mole

        # Update LJ parameters with perturbed parameters
        parmed.tools.changeLJSingleType(
            structure,
            f"@%{parameter_key.smirks}",
            lj_radius,
            lj_depth,
        ).execute()

        # NOTE: the code below saves the topology to file and load it back in
        # instead of creating the OpenMM system directly from the ParmEd object
        # due to a bug in ParmEd.
        working_directory = tempfile.mkdtemp()
        prmtop_path = os.path.join(
            working_directory,
            "modified_system.prmtop",
        )
        structure.save(prmtop_path)
        prmtop_file = openmm.app.AmberPrmtopFile(prmtop_path)

        gb_force = None
        nb_force = None
        for force in original_system.getForces():
            if isinstance(force, openmm.CustomGBForce) or isinstance(
                force, openmm.GBSAOBCForce
            ):
                gb_force = force
            elif isinstance(force, openmm.NonbondedForce):
                nb_force = force

        perturbed_system = prmtop_file.createSystem(
            nonbondedMethod=app.PME if gb_force is None else app.NoCutoff,
            nonbondedCutoff=nb_force.getCutoffDistance(),
            constraints=app.HBonds,
            rigidWater=True,
        )
        if enable_pbc and original_system.usesPeriodicBoundaryConditions():
            perturbed_system.setDefaultPeriodicBoxVectors(
                *original_system.getDefaultPeriodicBoxVectors()
            )

        shutil.rmtree(working_directory)

    elif parameter_key.tag == "GBSA":
        offset_factor = 0.009  # nm
        prmtop = app.AmberPrmtopFile(topology_path)
        all_bonds = app.internal._get_bonded_atom_list(prmtop.topology)

        # Get GB force object
        gbsa_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, openmm.CustomGBForce) or isinstance(
                force, openmm.GBSAOBCForce
            ):
                gbsa_force = force

        if gbsa_force is None:
            logger.info(
                "GBSA force not found in System object, returning an empty `system` object and "
                "setting `parameter_value` to zero."
            )
            empty_system = openmm.System()
            parameter_value = scale_amount if scale_amount > 0.0 else 0.0
            if parameter_key.attribute == "radius":
                parameter_value *= openmm_unit.nanometer
            elif parameter_key.attribute == "scale":
                parameter_value *= openmm_unit.dimensionless

            return empty_system, parameter_value

        # Determine element of atom
        mask_element = app.element.get_by_symbol(parameter_key.smirks[0])
        connect_element = None
        if "-" in parameter_key.smirks[0]:
            connect_element = app.element.get_by_symbol(
                parameter_key.smirks.split("-")[-1]
            )

        # Find atom in system to change GB radii
        for atom in prmtop.topology.atoms():
            current_atom = None
            element = atom.element

            if element is mask_element and connect_element is None:
                current_atom = atom

            elif element is mask_element and connect_element:
                bondeds = all_bonds[atom]
                if bondeds[0].element is connect_element:
                    current_atom = atom

            if current_atom:
                current_param = gbsa_force.getParticleParameters(current_atom.index)
                charge = current_param[0]
                GB_radii = current_param[1] + offset_factor
                GB_scale = current_param[2] / current_param[1]

                if parameter_key.attribute == "radius":
                    GB_radii *= 1.0 + scale_amount
                elif parameter_key.attribute == "scale":
                    GB_scale *= 1.0 + scale_amount

                offset_radii = GB_radii - offset_factor
                scaled_radii = offset_radii * GB_scale
                gbsa_force.setParticleParameters(
                    current_atom.index, [charge, offset_radii, scaled_radii]
                )

        # Convert parameter to a openmm.unit.Quantity
        if parameter_key.attribute == "radius":
            parameter_value = GB_radii * openmm_unit.nanometer
        elif parameter_key.attribute == "scale":
            parameter_value = GB_scale * openmm_unit.dimensionless

    else:
        raise ValueError(
            f"The parameter `{parameter_key.tag}` is not supported for GAFF gradient calculation."
        )

    return perturbed_system, parameter_value
