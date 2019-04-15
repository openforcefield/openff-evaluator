from propertyestimator.backends import DaskLSFBackend, QueueComputeResources


def test_dask_lsf_creation():
    """Test creating and starting a new dask LSF backend."""

    cpu_backend = DaskLSFBackend()
    cpu_backend.start()
    cpu_backend.stop()

    gpu_resources = QueueComputeResources(1, 1, preferred_gpu_toolkit=QueueComputeResources.GPUToolkit.CUDA)

    gpu_commands = [
        'module load cuda/9.2',
    ]

    gpu_backend = DaskLSFBackend(resources_per_worker=gpu_resources,
                                 queue_name='gpuqueue',
                                 extra_script_commands=gpu_commands)

    gpu_backend.start()
    gpu_backend.stop()
