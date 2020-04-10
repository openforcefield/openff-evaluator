#!/usr/bin/env python
import json
import os
import shutil

from openforcefield.typing.engines import smirnoff

from propertyestimator import unit
from propertyestimator.backends import QueueWorkerResources, DaskPBSBackend
from propertyestimator.client import PropertyEstimatorClient, PropertyEstimatorOptions
from propertyestimator.datasets.taproom import TaproomDataSet
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging, get_data_filename
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow import WorkflowOptions


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml',
                                      get_data_filename('forcefield/tip3p.offxml'))

    # Load in the data set, retaining only a specific host / guest pair.
    host = 'acd'
    guest = 'bam'

    data_set = TaproomDataSet()

    data_set.filter_by_host_identifiers(host)
    data_set.filter_by_guest_identifiers(guest)

    # Set up the server object which run the calculations.
    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if os.path.isdir(working_directory):
        shutil.rmtree(working_directory)

    queue_resources = QueueWorkerResources(number_of_threads=1,
                                           number_of_gpus=1,
                                           preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                           per_thread_memory_limit=4 * unit.gigabyte,
                                           wallclock_time_limit="05:59")

    setup_script_commands = [
        'source /home/davids4/.bashrc',
        'conda activate propertyestimator'
    ]

    calculation_backend = DaskPBSBackend(minimum_number_of_workers=1,
                                         maximum_number_of_workers=3,
                                         resources_per_worker=queue_resources,
                                         queue_name='home-mgilson',
                                         setup_script_commands=setup_script_commands,
                                         adaptive_interval='1000ms',
                                         resource_line='nodes=1:ppn=2')

    # Set up a backend to cache simulation data in.
    storage_backend = LocalFileStorage(storage_directory)

    # Spin up the server object.
    PropertyEstimatorServer(calculation_backend=calculation_backend,
                            storage_backend=storage_backend,
                            working_directory=working_directory)

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
