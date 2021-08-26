"""
A set of utilities for helping to perform simulations using openmm.
"""
import copy
import logging
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Optional, Tuple

import numpy
import parmed as pmd
from pint import UndefinedUnitError
from simtk import openmm
from simtk import unit as simtk_unit

from openff.evaluator import unit
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
    Returns
    -------
    Platform
        The created platform
    """
    from simtk.openmm import Platform

    # Setup the requested platform:
    if compute_resources.number_of_gpus > 0:

        # TODO: Make sure use mixing precision - CUDA, OpenCL.
        # TODO: Deterministic forces = True

        from openff.evaluator.backends import ComputeResources

        toolkit_enum = ComputeResources.GPUToolkit(
            compute_resources.preferred_gpu_toolkit
        )

        # A platform which runs on GPUs has been requested.
        platform_name = (
            "CUDA"
            if toolkit_enum == ComputeResources.GPUToolkit.CUDA
            else ComputeResources.GPUToolkit.OpenCL
        )

        # noinspection PyCallByClass,PyTypeChecker
        platform = Platform.getPlatformByName(platform_name)

        if compute_resources.gpu_device_indices is not None:

            property_platform_name = platform_name

            if toolkit_enum == ComputeResources.GPUToolkit.CUDA:
                property_platform_name = platform_name.lower().capitalize()

            platform.setPropertyDefaultValue(
                property_platform_name + "DeviceIndex",
                compute_resources.gpu_device_indices,
            )

        if high_precision:
            platform.setPropertyDefaultValue("Precision", "double")

        logger.info(
            "Setting up an openmm platform on GPU {}".format(
                compute_resources.gpu_device_indices or 0
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


def openmm_quantity_to_pint(openmm_quantity):
    """Converts a `simtk.unit.Quantity` to a `openff.evaluator.unit.Quantity`.

    Parameters
    ----------
    openmm_quantity: simtk.unit.Quantity
        The quantity to convert.

    Returns
    -------
    openff.evaluator.unit.Quantity
        The converted quantity.
    """

    if openmm_quantity is None or isinstance(openmm_quantity, UndefinedAttribute):
        return None

    assert isinstance(openmm_quantity, simtk_unit.Quantity)

    openmm_unit = openmm_quantity.unit
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    pint_unit = openmm_unit_to_pint(openmm_unit)
    pint_quantity = openmm_raw_value * pint_unit

    return pint_quantity


def openmm_unit_to_pint(openmm_unit):
    """Converts a `simtk.unit.Unit` to a `openff.evaluator.unit.Unit`.

    Parameters
    ----------
    openmm_unit: simtk.unit.Unit
        The unit to convert.

    Returns
    -------
    openff.evaluator.unit.Unit
        The converted unit.
    """
    from openff.toolkit.utils import unit_to_string

    if openmm_unit is None or isinstance(openmm_unit, UndefinedAttribute):
        return None

    assert isinstance(openmm_unit, simtk_unit.Unit)

    openmm_unit_string = unit_to_string(openmm_unit)

    # Handle the case whereby OMM treats daltons as having
    # units of g / mol, whereas SI and pint define them to
    # have units of kg.
    openmm_unit_string = (
        None
        if openmm_unit_string is None
        else openmm_unit_string.replace("dalton", "(gram / mole)")
    )

    try:
        pint_unit = unit(openmm_unit_string).units
    except UndefinedUnitError:

        logger.info(
            f"The {openmm_unit_string} OMM unit string (based on the {openmm_unit} object) "
            f"is not supported."
        )

        raise

    return pint_unit


def pint_quantity_to_openmm(pint_quantity):
    """Converts a `openff.evaluator.unit.Quantity` to a `simtk.unit.Quantity`.

    Notes
    -----
    Not all pint units are available in OpenMM.

    Parameters
    ----------
    pint_quantity: openff.evaluator.unit.Quantity
        The quantity to convert.

    Returns
    -------
    simtk.unit.Quantity
        The converted quantity.
    """

    if pint_quantity is None or isinstance(pint_quantity, UndefinedAttribute):
        return None

    assert isinstance(pint_quantity, unit.Quantity)

    pint_unit = pint_quantity.units
    pint_raw_value = pint_quantity.magnitude

    openmm_unit = pint_unit_to_openmm(pint_unit)
    openmm_quantity = pint_raw_value * openmm_unit

    return openmm_quantity


def pint_unit_to_openmm(pint_unit):
    """Converts a `openff.evaluator.unit.Unit` to a `simtk.unit.Unit`.

    Notes
    -----
    Not all pint units are available in OpenMM.

    Parameters
    ----------
    pint_unit: openff.evaluator.unit.Unit
        The unit to convert.

    Returns
    -------
    simtk.unit.Unit
        The converted unit.
    """
    from openff.toolkit.utils import string_to_unit

    if pint_unit is None or isinstance(pint_unit, UndefinedAttribute):
        return None

    assert isinstance(pint_unit, unit.Unit)

    pint_unit_string = f"{pint_unit:!s}"

    # Handle a unit name change in pint 0.10.*
    pint_unit_string = pint_unit_string.replace("standard_atmosphere", "atmosphere")

    try:
        # noinspection PyTypeChecker
        openmm_unit = string_to_unit(pint_unit_string)
    except AttributeError:

        logger.info(
            f"The {pint_unit_string} pint unit string (based on the {pint_unit} object) "
            f"could not be understood by `openff.toolkit.utils.string_to_unit`"
        )

        raise

    return openmm_unit


def disable_pbc(system):
    """Disables any periodic boundary conditions being applied
    to non-bonded forces by setting the non-bonded method to
    `NoCutoff = 0`

    Parameters
    ----------
    system: simtk.openmm.system
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
) -> Tuple["openmm.System", "simtk_unit.Quantity"]:
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
    # simtk units here.
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

    if parameter_key.tag in ["GBSA", "CustomGBSA"]:
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

    parameter = (
        handler
        if parameter_key.smirks is None
        else handler.parameters[parameter_key.smirks]
    )

    parameter_value = getattr(parameter, parameter_key.attribute)

    # Convert float values to unitless simtk.Quantity
    if not isinstance(parameter_value, simtk_unit.Quantity):
        parameter_value = parameter_value * simtk_unit.dimensionless

    # Optionally perturb the parameter of interest.
    if scale_amount is not None:

        parameter_unit = parameter_value.unit

        if numpy.isclose(parameter_value.value_in_unit(parameter_unit), 0.0):
            # Careful thought needs to be given to this. Consider cases such as
            # epsilon or sigma where negative values are not allowed.
            parameter_value = scale_amount if scale_amount > 0.0 else 0.0

            if parameter_unit != simtk_unit.dimensionless:
                parameter_value = parameter_value * parameter_unit

        else:
            parameter_value *= 1.0 + scale_amount

    setattr(parameter, parameter_key.attribute, parameter_value)

    if not isinstance(parameter_value, simtk_unit.Quantity):
        parameter_value = parameter_value * simtk_unit.dimensionless

    # Create the parameterized sub-system.
    system = force_field_subset.create_openmm_system(topology)

    return system, parameter_value


def perturbed_gaff_system(
    parameter_key: ParameterGradientKey,
    system_path: str,
    topology_path: str,
    enable_pbc: bool,
    scale_amount: Optional[float] = None,
) -> Tuple["openmm.System", "simtk_unit.Quantity"]:
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

    from simtk.openmm import System as OMMSystem

    if not os.path.isfile(topology_path):
        logger.info(
            "GAFF topology file not found, returning an empty `system` object and "
            "setting `parameter_value` to zero."
        )

        perturbed_system = OMMSystem()
        parameter_value = 0.0

        if scale_amount is not None:
            parameter_value = scale_amount if scale_amount > 0.0 else 0.0

        if parameter_key.tag == "Bond":
            if parameter_key.attribute == "length":
                parameter_value *= simtk_unit.angstrom
            elif parameter_key.attribute == "k":
                parameter_value *= (
                    simtk_unit.kilocalorie_per_mole / simtk_unit.angstrom ** 2
                )

        elif parameter_key.tag == "Angle":
            if parameter_key.attribute == "theta":
                parameter_value *= simtk_unit.degree
            elif parameter_key.attribute == "k":
                parameter_value *= (
                    simtk_unit.kilocalorie_per_mole / simtk_unit.radians ** 2
                )

        elif parameter_key.tag == "vdW":
            if parameter_key.attribute == "rmin_half":
                parameter_value *= simtk_unit.angstrom
            elif parameter_key.attribute == "epsilon":
                parameter_value *= simtk_unit.kilocalorie_per_mole

        elif parameter_key.tag == "GBSA":
            if parameter_key.attribute == "radius":
                parameter_value *= simtk_unit.nanometer
            elif parameter_key.attribute == "scale":
                parameter_value *= simtk_unit.dimensionless

        return perturbed_system, parameter_value

    # Load Topology and System XML
    structure = pmd.load_file(topology_path)
    with open(system_path, "r") as f:
        original_system = openmm.XmlSerializer.deserialize(f.read())
    perturbed_system = copy.deepcopy(original_system)

    if parameter_key.tag == "Bond":

        from simtk.openmm import HarmonicBondForce

        # Bond atom types
        bond_type = parameter_key.smirks.replace("-", " ").split()
        bond_list = []

        # Get Bond parameters
        bonds = str(pmd.tools.printBonds(structure)).split("\n")
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
            rbond *= (1.0 + scale_amount) * simtk_unit.angstrom
            parameter_value = rbond
        elif parameter_key.attribute == "k":
            kbond *= (
                (1.0 + scale_amount)
                * simtk_unit.kilocalorie_per_mole
                / simtk_unit.angstrom ** 2
            )
            parameter_value = kbond

        # Get bond force from system
        bond_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, HarmonicBondForce):
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

        from simtk.openmm import HarmonicAngleForce

        # Angle atom types
        angle_type = parameter_key.smirks.replace("-", " ").split()
        angle_list = []

        # Get Angle parameters
        angles = str(pmd.tools.printAngles(structure)).split("\n")
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
            theta *= (1.0 + scale_amount) * simtk_unit.degree
            parameter_value = theta
        elif parameter_key.attribute == "k":
            kangle *= (
                (1.0 + scale_amount)
                * simtk_unit.kilocalorie_per_mole
                / simtk_unit.radians ** 2
            )
            parameter_value = kangle

        # Get angle force from system
        angle_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, HarmonicAngleForce):
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

        from simtk.openmm import CustomGBForce, GBSAOBCForce, NonbondedForce

        # Get current LJ parameters
        lj_index = structure.LJ_types[parameter_key.smirks] - 1
        lj_radius = structure.LJ_radius[lj_index]
        lj_depth = structure.LJ_depth[lj_index]

        # Determine which parameter to perturb
        if parameter_key.attribute == "rmin_half":
            lj_radius *= 1.0 + scale_amount
            parameter_value = lj_radius * simtk_unit.angstrom
        elif parameter_key.attribute == "epsilon":
            lj_depth *= 1.0 + scale_amount
            parameter_value = lj_depth * simtk_unit.kilocalorie_per_mole

        # Update LJ parameters with perturbed parameters
        pmd.tools.changeLJSingleType(
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
            if isinstance(force, CustomGBForce) or isinstance(force, GBSAOBCForce):
                gb_force = force
            elif isinstance(force, NonbondedForce):
                nb_force = force

        perturbed_system = prmtop_file.createSystem(
            nonbondedMethod=openmm.app.PME if gb_force is None else openmm.app.NoCutoff,
            nonbondedCutoff=nb_force.getCutoffDistance(),
            constraints=openmm.app.HBonds,
            rigidWater=True,
        )
        if enable_pbc and original_system.usesPeriodicBoundaryConditions():
            perturbed_system.setDefaultPeriodicBoxVectors(
                *original_system.getDefaultPeriodicBoxVectors()
            )

        shutil.rmtree(working_directory)

    elif parameter_key.tag == "GBSA":

        from simtk.openmm import CustomGBForce, GBSAOBCForce
        from simtk.openmm.app import element as E
        from simtk.openmm.app.internal.customgbforces import _get_bonded_atom_list

        offset_factor = 0.009  # nm
        prmtop = openmm.app.AmberPrmtopFile(topology_path)
        all_bonds = _get_bonded_atom_list(prmtop.topology)

        # Get GB force object
        gbsa_force = None
        for force in perturbed_system.getForces():
            if isinstance(force, CustomGBForce) or isinstance(force, GBSAOBCForce):
                gbsa_force = force

        if gbsa_force is None:
            logger.info(
                "GBSA force not found in System object, returning an empty `system` object and "
                "setting `parameter_value` to zero."
            )
            empty_system = OMMSystem()
            parameter_value = scale_amount if scale_amount > 0.0 else 0.0
            if parameter_key.attribute == "radius":
                parameter_value *= simtk_unit.nanometer
            elif parameter_key.attribute == "scale":
                parameter_value *= simtk_unit.dimensionless

            return empty_system, parameter_value

        # Determine element of atom
        mask_element = E.get_by_symbol(parameter_key.smirks[0])
        connect_element = None
        if "-" in parameter_key.smirks[0]:
            connect_element = E.get_by_symbol(parameter_key.smirks.split("-")[-1])

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

        # Convert parameter to a simtk.unit.Quantity
        if parameter_key.attribute == "radius":
            parameter_value = GB_radii * simtk_unit.nanometer
        elif parameter_key.attribute == "scale":
            parameter_value = GB_scale * simtk_unit.dimensionless

    else:

        raise ValueError(
            f"The parameter `{parameter_key.tag}` is not supported for GAFF gradient calculation."
        )

    return perturbed_system, parameter_value
