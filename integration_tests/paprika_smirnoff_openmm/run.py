#!/usr/bin/env python
import json
import os
import shutil
from enum import Enum

from openforcefield.typing.engines import smirnoff
from propertyestimator import unit, server
from propertyestimator.backends import DaskLocalCluster, ComputeResources, QueueWorkerResources, DaskLSFBackend
from propertyestimator.client import PropertyEstimatorClient, PropertyEstimatorOptions
from propertyestimator.datasets.taproom import TaproomDataSet
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging, get_data_filename
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow import WorkflowOptions


class BackendType(Enum):
    LocalCPU = 'LocalCPU'
    LocalGPU = 'LocalGPU'
    GPU = 'GPU'
    CPU = 'CPU'


def setup_server(backend_type=BackendType.LocalCPU, max_number_of_workers=1,
                 conda_environment='propertyestimator', worker_memory=8 * unit.gigabyte, port=8000):

    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if os.path.isdir(working_directory):
        shutil.rmtree(working_directory)

    calculation_backend = None

    if backend_type == BackendType.LocalCPU:
        calculation_backend = DaskLocalCluster(number_of_workers=1)

    elif backend_type == BackendType.LocalGPU:

        calculation_backend = DaskLocalCluster(number_of_workers=1,
                                                      resources_per_worker=ComputeResources(1,
                                                                                            1,
                                                                                            ComputeResources.
                                                                                            GPUToolkit.CUDA))

    elif backend_type == BackendType.GPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               number_of_gpus=1,
                                               preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                               per_thread_memory_limit=worker_memory,
                                               wallclock_time_limit="05:59")

        worker_script_commands = [
            f'export OE_LICENSE="/home/boothros/oe_license.txt"',
            f'. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            f'conda activate {conda_environment}',
            f'module load cuda/9.2'
        ]

        extra_script_options = [
            '-m "ls-gpu lt-gpu"'
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='gpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             extra_script_options=extra_script_options,
                                             adaptive_interval='1000ms')
    elif backend_type == BackendType.CPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=worker_memory,
                                               wallclock_time_limit="01:30")

        worker_script_commands = [
            f'export OE_LICENSE="/home/boothros/oe_license.txt"',
            f'. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            f'conda activate {conda_environment}',
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='cpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             adaptive_interval='1000ms')

    storage_backend = LocalFileStorage(storage_directory)

    server.PropertyEstimatorServer(calculation_backend=calculation_backend,
                                   storage_backend=storage_backend,
                                   port=port,
                                   working_directory=working_directory)


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml',
                                      get_data_filename('forcefield/tip3p.offxml'))

    # Load in the data set.
    host = 'acd'
    guest = 'bam'

    data_set = TaproomDataSet(default_ionic_strength=None)

    data_set.filter_by_host_identifiers(host)
    data_set.filter_by_guest_identifiers(guest)

    # Set up the server object which run the calculations.
    setup_server(backend_type=BackendType.LocalCPU,
                 max_number_of_workers=1,
                 worker_memory=3.5 * unit.gigabyte)

    # Request the estimate of the host-guest binding affinity.
    options = PropertyEstimatorOptions()
    options.allowed_calculation_layers = ['SimulationLayer']

    options.workflow_options = {
        'HostGuestBindingAffinity': {
            'SimulationLayer': WorkflowOptions(convergence_mode=WorkflowOptions.ConvergenceMode.NoChecks),
        }
    }

    estimator_client = PropertyEstimatorClient()

    request = estimator_client.request_estimate(property_set=data_set,
                                                force_field_source=force_field,
                                                options=options)

    # Wait for the results.
    results = request.results(True, 30)

    # Save the result to file.
    with open('results.json', 'wb') as file:

        json_results = json.dumps(results, sort_keys=True, indent=2,
                                  separators=(',', ': '), cls=TypedJSONEncoder)

        file.write(json_results.encode('utf-8'))


if __name__ == "__main__":
    main()
