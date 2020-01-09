from enum import Enum

from propertyestimator import server, unit
from propertyestimator.backends import (
    ComputeResources,
    DaskLocalCluster,
    DaskLSFBackend,
    QueueWorkerResources,
)


class BackendType(Enum):
    LocalCPU = "LocalCPU"
    LocalGPU = "LocalGPU"
    GPU = "GPU"
    CPU = "CPU"


def setup_server(
    backend_type=BackendType.LocalCPU,
    max_number_of_workers=1,
    conda_environment="propertyestimator",
    worker_memory=4 * unit.gigabyte,
    port=8000,
    cuda_version="10.1",
):
    """A convenience function to sets up an estimation server which will can advantage
    of different compute backends.

    Parameters
    ----------
    backend_type: BackendType
        The type of compute backend to use.
    max_number_of_workers: int
        The maximum number of workers to adaptively insert into
        the queuing system.
    conda_environment: str
        The name of the conda environment in which the propertyestimator
        package is installed.
    worker_memory: pint.Quantity
        The maximum amount of memory to request per worker.
    port: int
        The port that the server should listen for estimation requests on.
    cuda_version: str
        The version of CUDA to use if running on a backend which supports
        GPUs.

    Returns
    -------
    EvaluatorServer
        The server object.
    """

    calculation_backend = None

    if backend_type == BackendType.LocalCPU:
        calculation_backend = DaskLocalCluster(number_of_workers=max_number_of_workers)

    elif backend_type == BackendType.LocalGPU:

        calculation_backend = DaskLocalCluster(
            number_of_workers=max_number_of_workers,
            resources_per_worker=ComputeResources(
                1, 1, ComputeResources.GPUToolkit.CUDA
            ),
        )

    elif backend_type == BackendType.GPU:

        queue_resources = QueueWorkerResources(
            number_of_threads=1,
            number_of_gpus=1,
            preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
            per_thread_memory_limit=worker_memory,
            wallclock_time_limit="05:59",
        )

        worker_script_commands = [
            f"conda activate {conda_environment}",
            f"module load cuda/{cuda_version}",
        ]

        calculation_backend = DaskLSFBackend(
            minimum_number_of_workers=1,
            maximum_number_of_workers=max_number_of_workers,
            resources_per_worker=queue_resources,
            queue_name="gpuqueue",
            setup_script_commands=worker_script_commands,
            adaptive_interval="1000ms",
        )
    elif backend_type == BackendType.CPU:

        queue_resources = QueueWorkerResources(
            number_of_threads=1,
            per_thread_memory_limit=worker_memory,
            wallclock_time_limit="01:30",
        )

        worker_script_commands = [f"conda activate {conda_environment}"]

        calculation_backend = DaskLSFBackend(
            minimum_number_of_workers=1,
            maximum_number_of_workers=max_number_of_workers,
            resources_per_worker=queue_resources,
            queue_name="cpuqueue",
            setup_script_commands=worker_script_commands,
            adaptive_interval="1000ms",
        )

    calculation_backend.start()

    # Spin up the server object.
    return server.EvaluatorServer(calculation_backend=calculation_backend, port=port)
