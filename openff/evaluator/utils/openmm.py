"""
A set of utilities for helping to perform simulations using openmm.
"""
import copy
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy

from openff.evaluator.attributes.attributes import UndefinedAttribute

try:
    import openmm
    from openmm import app
    from openmm import unit as _openmm_unit
except ImportError:
    from simtk import openmm
    from simtk.openmm import app
    from simtk.openmm import unit as _openmm_unit

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
    try:
        from openmm import Platform
    except ImportError:
        from simtk.openmm import Platform

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
) -> Tuple["openmm.System", "_openmm_unit.Quantity"]:
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
                "epsilon": 0.0 * _openmm_unit.kilocalories_per_mole,
                "sigma": 1.0 * _openmm_unit.angstrom,
            }
        )

    parameter = (
        handler
        if parameter_key.smirks is None
        else handler.parameters[parameter_key.smirks]
    )

    parameter_value = getattr(parameter, parameter_key.attribute)
    is_quantity = isinstance(parameter_value, _openmm_unit.Quantity)

    if not is_quantity:
        parameter_value = parameter_value * _openmm_unit.dimensionless

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

    if not isinstance(parameter_value, _openmm_unit.Quantity):
        # Handle the case where OMM down-converts a dimensionless quantity to a float.
        parameter_value = parameter_value * _openmm_unit.dimensionless

    setattr(
        parameter,
        parameter_key.attribute,
        parameter_value
        if is_quantity
        else parameter_value.value_in_unit(_openmm_unit.dimensionless),
    )

    # Create the parameterized sub-system.
    system = force_field_subset.create_openmm_system(topology)
    return system, parameter_value


def update_context_with_positions(
    context: openmm.Context,
    positions: _openmm_unit.Quantity,
    box_vectors: Optional[_openmm_unit.Quantity],
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
                new_positions[j] = positions[i].value_in_unit(_openmm_unit.nanometers)
                i += 1

        positions = new_positions * _openmm_unit.nanometers

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
) -> _openmm_unit.Quantity:
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

    from openff.units.openmm import from_openmm

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
