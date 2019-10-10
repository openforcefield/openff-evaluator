#!/usr/bin/env python
import json

from openforcefield.typing.engines import smirnoff

from integration_tests.utils import setup_server, BackendType
from propertyestimator.client import PropertyEstimatorClient, PropertyEstimatorOptions
from propertyestimator.datasets.taproom import TaproomDataSet
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
    setup_server(backend_type=BackendType.LocalGPU, max_number_of_workers=1)

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
