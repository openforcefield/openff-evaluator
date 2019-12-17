#!/usr/bin/env python3
import shutil
from enum import Enum
from os import path

from propertyestimator import unit
from propertyestimator.backends import (
    ComputeResources,
    DaskLocalCluster,
    DaskLSFBackend,
    QueueWorkerResources,
)
from propertyestimator.server.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging


class BackendType(Enum):

    LocalCPU = "LocalCPU"
    LocalGPU = "LocalGPU"
    LilacGPU = "LilacGPU"
    LilacCPU = "LilacCPU"


def setup_server(backend_type=BackendType.LocalCPU, max_number_of_workers=1):
    """Creates a new estimation server object.

    Parameters
    ----------
    backend_type: BackendType
        The type of backend to use.
    max_number_of_workers: int
        The maximum number of compute workers to spin up.

    Returns
    -------
    PropertyEstimatorServer
        The server object.
    """

    # Set the name of the directory in which all temporary files
    # will be generated.
    working_directory = "working_directory"

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Set the name of the directory in which all cached simulation data
    # will be stored..
    storage_directory = "storage_directory"

    # Set up the backend which will perform any calculations.
    calculation_backend = None

    if backend_type == BackendType.LocalCPU:

        # A backend which will run all calculations on the local machines CPUs.
        calculation_backend = DaskLocalCluster(number_of_workers=max_number_of_workers)

    if backend_type == BackendType.LocalGPU:

        # A backend which will run all calculations on the local machines GPUs.
        compute_resources = ComputeResources(number_of_threads=1, number_of_gpus=1)

        calculation_backend = DaskLocalCluster(
            number_of_workers=max_number_of_workers,
            resources_per_worker=compute_resources,
        )

    elif backend_type == BackendType.LilacGPU:

        # A backend which will run all calculations on the MSKCC `lilac` cluster, taking
        # advantage of the available GPUs.
        queue_resources = QueueWorkerResources(
            number_of_threads=1,
            number_of_gpus=1,
            preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
            per_thread_memory_limit=5 * unit.gigabyte,
            wallclock_time_limit="05:59",
        )

        extra_script_options = ['-m "ls-gpu lt-gpu"']

        worker_script_commands = [
            'export OE_LICENSE="/home/boothros/oe_license.txt"',
            ". /home/boothros/miniconda3/etc/profile.d/conda.sh",
            "conda activate forcebalance",
            "module load cuda/9.2",
        ]

        calculation_backend = DaskLSFBackend(
            minimum_number_of_workers=1,
            maximum_number_of_workers=max_number_of_workers,
            resources_per_worker=queue_resources,
            queue_name="gpuqueue",
            setup_script_commands=worker_script_commands,
            extra_script_options=extra_script_options,
            adaptive_interval="1000ms",
        )

    elif backend_type == BackendType.LilacCPU:

        # A backend which will run all calculations on the MSKCC `lilac` cluster using onlu
        # CPUs.
        queue_resources = QueueWorkerResources(
            number_of_threads=1,
            per_thread_memory_limit=5 * unit.gigabyte,
            wallclock_time_limit="01:30",
        )

        worker_script_commands = [
            'export OE_LICENSE="/home/boothros/oe_license.txt"',
            ". /home/boothros/miniconda3/etc/profile.d/conda.sh",
            "conda activate forcebalance",
        ]

        calculation_backend = DaskLSFBackend(
            minimum_number_of_workers=1,
            maximum_number_of_workers=max_number_of_workers,
            resources_per_worker=queue_resources,
            queue_name="cpuqueue",
            setup_script_commands=worker_script_commands,
            adaptive_interval="1000ms",
        )

    # Set up the storage backend.
    storage_backend = LocalFileStorage(storage_directory)

    # Set up the server itself.
    server = PropertyEstimatorServer(
        calculation_backend=calculation_backend,
        storage_backend=storage_backend,
        working_directory=working_directory,
    )

    return server


def main():

    # Setup logging to a local file.
    setup_timestamp_logging("server_logger_output.log")

    # Create the server.
    server = setup_server(backend_type=BackendType.LocalCPU, max_number_of_workers=1)

    # Tell the server to start listening for incoming
    # estimation requests.
    server.start_listening_loop()


if __name__ == "__main__":
    main()
