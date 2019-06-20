"""
A set of utilities for helping to perform simulations using openmm.
"""
import logging


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

        # noinspection PyCallByClass,PyTypeChecker
        if not high_precision:
            platform = Platform.getPlatformByName('CPU')
            platform.setPropertyDefaultValue('Threads', str(compute_resources.number_of_threads))
        else:
            platform = Platform.getPlatformByName('Reference')

        logging.info('Setting up a simulation with {} threads'.format(compute_resources.number_of_threads))

    return platform
