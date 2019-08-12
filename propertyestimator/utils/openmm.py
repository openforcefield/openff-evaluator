"""
A set of utilities for helping to perform simulations using openmm.
"""
import logging
import os

from pint import UndefinedUnitError

from propertyestimator import unit
from simtk import unit as simtk_unit


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

        from propertyestimator.backends import ComputeResources
        toolkit_enum = ComputeResources.GPUToolkit(compute_resources.preferred_gpu_toolkit)

        # A platform which runs on GPUs has been requested.
        platform_name = 'CUDA' if toolkit_enum == ComputeResources.GPUToolkit.CUDA else \
                                                  ComputeResources.GPUToolkit.OpenCL

        # noinspection PyCallByClass,PyTypeChecker
        platform = Platform.getPlatformByName(platform_name)

        if compute_resources.gpu_device_indices is not None:

            property_platform_name = platform_name

            if toolkit_enum == ComputeResources.GPUToolkit.CUDA:
                property_platform_name = platform_name.lower().capitalize()

            platform.setPropertyDefaultValue(property_platform_name + 'DeviceIndex',
                                             compute_resources.gpu_device_indices)

        if high_precision:
            platform.setPropertyDefaultValue('Precision', 'double')

        logging.info('Setting up an openmm platform on GPU {}'.format(compute_resources.gpu_device_indices or 0))

    else:

        if not high_precision:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName('CPU')
            platform.setPropertyDefaultValue('Threads', str(compute_resources.number_of_threads))
        else:
            # noinspection PyCallByClass,PyTypeChecker
            platform = Platform.getPlatformByName('Reference')

        logging.info('Setting up a simulation with {} threads'.format(compute_resources.number_of_threads))

    return platform


# Some openmm units are not currently supported.
unsupported_openmm_units = {
    simtk_unit.yottojoule,
    simtk_unit.item,
    simtk_unit.yottopascal,
    simtk_unit.century,
    simtk_unit.yottosecond,
    simtk_unit.yottogram,
    simtk_unit.bohr,
    simtk_unit.yottocalorie,
    simtk_unit.yottoliter,
    simtk_unit.yottometer,
    simtk_unit.debye,
    simtk_unit.yottonewton,
    simtk_unit.ban,
    simtk_unit.yottomolar,
    simtk_unit.nat,
    simtk_unit.mmHg,
    simtk_unit.year,
    simtk_unit.psi,
    simtk_unit.pound_mass,
    simtk_unit.stone,
    simtk_unit.millenium
}


def openmm_quantity_to_pint(openmm_quantity):
    """Converts a `simtk.unit.Quantity` to a `pint.Quantity`.

    Parameters
    ----------
    openmm_quantity: simtk.unit.Quantity
        The quantity to convert.

    Returns
    -------
    pint.Quantity
        The converted quantity.
    """

    if openmm_quantity is None:
        return None

    assert isinstance(openmm_quantity, simtk_unit.Quantity)

    if openmm_quantity.unit in unsupported_openmm_units:

        raise ValueError(f'Quantities bearing the {openmm_quantity.unit} are not '
                         f'currently supported by pint.')

    openmm_unit = openmm_quantity.unit
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    pint_unit = openmm_unit_to_pint(openmm_unit)
    pint_quantity = openmm_raw_value * pint_unit

    return pint_quantity


def openmm_unit_to_pint(openmm_unit):
    """Converts a `simtk.unit.Unit` to a `pint.Unit`.

    Parameters
    ----------
    openmm_unit: simtk.unit.Unit
        The unit to convert.

    Returns
    -------
    pint.Unit
        The converted unit.
    """
    from openforcefield.utils import unit_to_string

    if openmm_unit is None:
        return None

    assert isinstance(openmm_unit, simtk_unit.Unit)

    if openmm_unit in unsupported_openmm_units:

        raise ValueError(f'Quantities bearing the {openmm_unit} are not '
                         f'currently supported by pint.')

    openmm_unit_string = unit_to_string(openmm_unit)

    # Handle the case whereby OMM treats daltons as having
    # units of g / mol, whereas SI and pint define them to
    # have units of kg.
    openmm_unit_string = (None if openmm_unit_string is None else
                          openmm_unit_string.replace('dalton', '(gram / mole)'))

    try:
        pint_unit = unit(openmm_unit_string).units
    except UndefinedUnitError:

        logging.info(f'The {openmm_unit_string} OMM unit string (based on the {openmm_unit} object) '
                     f'is undefined in pint')

        raise

    return pint_unit


def pint_quantity_to_openmm(pint_quantity):
    """Converts a `pint.Quantity` to a `simtk.unit.Quantity`.

    Notes
    -----
    Not all pint units are available in OpenMM.

    Parameters
    ----------
    pint_quantity: pint.Quantity
        The quantity to convert.

    Returns
    -------
    simtk.unit.Quantity
        The converted quantity.
    """

    if pint_quantity is None:
        return None

    assert isinstance(pint_quantity, unit.Quantity)

    pint_unit = pint_quantity.units
    pint_raw_value = pint_quantity.magnitude

    openmm_unit = pint_unit_to_openmm(pint_unit)
    openmm_quantity = pint_raw_value * openmm_unit

    return openmm_quantity


def pint_unit_to_openmm(pint_unit):
    """Converts a `pint.Unit` to a `simtk.unit.Unit`.

    Notes
    -----
    Not all pint units are available in OpenMM.

    Parameters
    ----------
    pint_unit: pint.Unit
        The unit to convert.

    Returns
    -------
    simtk.unit.Unit
        The converted unit.
    """
    from openforcefield.utils import string_to_unit

    if pint_unit is None:
        return None

    assert isinstance(pint_unit, unit.Unit)

    pint_unit_string = str(pint_unit)

    try:
        # noinspection PyTypeChecker
        openmm_unit = string_to_unit(pint_unit_string)
    except AttributeError:

        logging.info(f'The {pint_unit_string} pint unit string (based on the {pint_unit} object) '
                     f'could not be understood by `openforcefield.utils.string_to_unit`')

        raise

    return openmm_unit


class BufferedFileObject:
    """A wrapper around a file object whose flush method
    does not do anything. The sole purpose of this class is
    to stop the OpenMM DCDFile reporter flushing to file
    after each step, which may be a bottle neck if the file
    is to be written to very frequently (e.g. when simulating
    a gas an saving many frames.
    """

    def __init__(self, file_object):
        self._file_object = file_object

    def seek(self, *args, **kwargs):
        self._file_object.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self._file_object.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self._file_object.write(*args, **kwargs)

    def flush(self):
        pass


class StateReporter:
    """StateReporter saves periodic checkpoints of a simulation context's
    state.

    This is intended to be a cross-hardware replacement for the built-in
    `simtk.openmm.CheckpointReporter`.

    Notes
    -----
    This class is entirely based on the `simtk.openmm.CheckpointReporter`
    class.
    """
    def __init__(self, file_path, report_interval):
        """Create a new `StateReporter` object.

        Parameters
        ----------
        file_path : str
            The path to the file to write to.
        report_interval : int
            The interval (in steps) at which to write the state of the system.
        """

        self._report_interval = report_interval
        self._file_path = file_path

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, True, True, True, True

    def report(self, _, state):
        """Generate a report.

        Parameters
        ----------
        state : State
            The current state of the simulation
        """
        from simtk.openmm import XmlSerializer

        # Serialize the state
        state_xml = XmlSerializer.serialize(state)

        # Attempt to do a thread safe write.
        file_path = self._file_path + '.tmp'

        with open(file_path, 'w') as file:
            file.write(state_xml)

        os.replace(file_path, self._file_path)
