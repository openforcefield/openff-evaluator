"""
A set of utilities for helping to perform simulations using openmm.
"""
import logging


def setup_platform_with_resources(compute_resources):
    """Creates an OpenMM `Platform` object which requests a set
    amount of compute resources (e.g with a certain number of cpus).

    Parameters
    ----------
    compute_resources: ComputeResources

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

            if compute_resources.preferred_gpu_toolkit == 'CUDA':
                property_platform_name = platform_name.lower().capitalize()

            platform.setPropertyDefaultValue(property_platform_name + 'DeviceIndex',
                                             compute_resources.gpu_device_indices)

        logging.info('Setting up an openmm platform on GPU {}'.format(compute_resources.gpu_device_indices or 0))

    else:

        # noinspection PyCallByClass,PyTypeChecker
        platform = Platform.getPlatformByName('CPU')
        platform.setPropertyDefaultValue('Threads', str(compute_resources.number_of_threads))

        logging.info('Setting up a simulation with {} threads'.format(compute_resources.number_of_threads))

    return platform
