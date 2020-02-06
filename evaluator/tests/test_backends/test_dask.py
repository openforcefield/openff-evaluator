import pytest

from evaluator.backends import DaskLSFBackend, DaskPBSBackend, QueueWorkerResources
from evaluator.backends.dask import _Multiprocessor
from evaluator.workflow.plugins import registered_workflow_protocols


def dummy_function(*args, **kwargs):

    assert len(args) == 1
    return args[0]


def test_dask_job_script_creation():
    """Test creating and starting a new dask LSF backend."""

    cpu_backend = DaskLSFBackend()
    cpu_backend.start()
    assert cpu_backend.job_script() is not None
    cpu_backend.stop()


@pytest.mark.parametrize("cluster_class", [DaskLSFBackend, DaskPBSBackend])
def test_dask_jobqueue_backend_creation(cluster_class):
    """Test creating and starting a new dask jobqueue backend."""

    cpu_backend = cluster_class()
    cpu_backend.start()
    cpu_backend.stop()

    gpu_resources = QueueWorkerResources(
        1, 1, preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA
    )

    gpu_commands = [
        "module load cuda/9.2",
    ]

    gpu_backend = cluster_class(
        resources_per_worker=gpu_resources,
        queue_name="gpuqueue",
        setup_script_commands=gpu_commands,
    )

    gpu_backend.start()
    assert "module load cuda/9.2" in gpu_backend.job_script()
    gpu_backend.stop()


@pytest.mark.skip(reason="This code currently hangs only on travis.")
def test_multiprocessor():

    expected_output = 12345

    return_value = _Multiprocessor.run(dummy_function, expected_output)
    assert expected_output == return_value


@pytest.mark.skip(reason="This code currently hangs only on travis.")
def test_lsf_wrapped_function():

    available_resources = QueueWorkerResources()

    protocols_to_import = [
        protocol_class.__module__ + "." + protocol_class.__qualname__
        for protocol_class in registered_workflow_protocols.values()
    ]

    per_worker_logging = True

    gpu_assignments = None

    expected_output = 12345

    result = DaskLSFBackend._wrapped_function(
        dummy_function,
        expected_output,
        available_resources=available_resources,
        registered_workflow_protocols=protocols_to_import,
        per_worker_logging=per_worker_logging,
        gpu_assignments=gpu_assignments,
    )

    assert expected_output == result
