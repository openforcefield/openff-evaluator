"""
A set of utilities for helping to perform simulations using openmm.
"""
import copy
import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy
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

        if not isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)):
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
    force_field_subset = ForceField()

    handlers_to_register = {parameter_key.tag}

    if parameter_key.tag in {"ChargeIncrementModel", "LibraryCharges"}:
        # Make sure to retain all of the electrostatic handlers when dealing with
        # charges as the applied charges will depend on which charges have been applied
        # by previous handlers.
        handlers_to_register.update(
            {"Electrostatics", "ChargeIncrementModel", "LibraryCharges"}
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
                "epsilon": 0.0 * simtk_unit.kilocalories_per_mole,
                "sigma": 1.0 * simtk_unit.angstrom,
            }
        )

    parameter = (
        handler
        if parameter_key.smirks is None
        else handler.parameters[parameter_key.smirks]
    )

    parameter_value = getattr(parameter, parameter_key.attribute)
    is_quantity = isinstance(parameter_value, simtk_unit.Quantity)

    if not is_quantity:
        parameter_value = parameter_value * simtk_unit.dimensionless

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

    if not isinstance(parameter_value, simtk_unit.Quantity):
        # Handle the case where OMM down-converts a dimensionless quantity to a float.
        parameter_value = parameter_value * simtk_unit.dimensionless

    setattr(
        parameter,
        parameter_key.attribute,
        parameter_value
        if is_quantity
        else parameter_value.value_in_unit(simtk_unit.dimensionless),
    )

    # Create the parameterized sub-system.
    system = force_field_subset.create_openmm_system(topology)
    return system, parameter_value
