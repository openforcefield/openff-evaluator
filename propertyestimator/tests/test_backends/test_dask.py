from propertyestimator.backends import DaskLSFBackend, QueueWorkerResources
from propertyestimator.backends.dask import Multiprocessor
from propertyestimator.workflow.plugins import available_protocols


def dummy_function(*args, **kwargs):

    assert len(args) == 1
    return args[0]


def test_dask_lsf_creation():
    """Test creating and starting a new dask LSF backend."""

    cpu_backend = DaskLSFBackend()
    cpu_backend.start()
    cpu_backend.stop()

    gpu_resources = QueueWorkerResources(1, 1, preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA)

    gpu_commands = [
        'module load cuda/9.2',
    ]

    gpu_backend = DaskLSFBackend(resources_per_worker=gpu_resources,
                                 queue_name='gpuqueue',
                                 setup_script_commands=gpu_commands)

    gpu_backend.start()
    gpu_backend.stop()


def test_multiprocessor():

    expected_output = 12345

    return_value = Multiprocessor.run(dummy_function, expected_output)
    assert expected_output == return_value


def test_lsf_wrapped_function():

    available_resources = QueueWorkerResources()

    protocols_to_import = [protocol_class.__module__ + '.' +
                           protocol_class.__qualname__ for protocol_class in available_protocols.values()]

    per_worker_logging = True

    gpu_assignments = None

    expected_output = 12345

    result = DaskLSFBackend._wrapped_function(dummy_function,
                                              expected_output,
                                              available_resources=available_resources,
                                              available_protocols=protocols_to_import,
                                              per_worker_logging=per_worker_logging,
                                              gpu_assignments=gpu_assignments)

    assert expected_output == result