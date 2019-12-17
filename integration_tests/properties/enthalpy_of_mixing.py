import json

from propertyestimator import client
from propertyestimator.client import RequestOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.properties import ParameterGradientKey
from integration_tests.utils import setup_server, BackendType
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow import WorkflowOptions


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_path = 'smirnoff99Frosst-1.1.0.offxml'
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Load in the data set containing a single enthalpy of mixing
    # property.
    with open('binary_data_set.json') as file:
        data_set = PhysicalPropertyDataSet.parse_json(file.read())

    data_set.filter_by_property_types('EnthalpyOfMixing')

    # Set up the server object which run the calculations.
    setup_server(backend_type=BackendType.LocalGPU, max_number_of_workers=1, port=8002)

    # Request the estimates.
    property_estimator = client.EvaluatorClient(client.ConnectionOptions(server_port=8002))

    options = RequestOptions()
    options.allowed_calculation_layers = ['SimulationLayer']

    options.workflow_options = {
        'EnthalpyOfMixing': {
            'SimulationLayer': WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks),
            'ReweightingLayer': WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks)
        }
    }

    parameter_gradient_keys = [
        ParameterGradientKey(tag='vdW', attribute='epsilon', smirks='[#6X4:1]'),
        ParameterGradientKey(tag='vdW', attribute='rmin_half', smirks='[#6X4:1]')
    ]

    request = property_estimator.request_estimate(property_set=data_set,
                                                  force_field_source=force_field_source,
                                                  options=options,
                                                  parameter_gradient_keys=parameter_gradient_keys)

    # Wait for the results.
    results = request.results(True, 5)

    # Save the result to file.
    with open('enthalpy_of_mixing_simulation.json', 'wb') as file:

        json_results = json.dumps(results, sort_keys=True, indent=2,
                                  separators=(',', ': '), cls=TypedJSONEncoder)

        file.write(json_results.encode('utf-8'))

    # Attempt to reweight the cached data.
    options.allowed_calculation_layers = ['ReweightingLayer']

    request = property_estimator.request_estimate(property_set=data_set,
                                                  force_field_source=force_field_source,
                                                  options=options,
                                                  parameter_gradient_keys=parameter_gradient_keys)

    # Wait for the results.
    results = request.results(True, 5)

    # Save the result to file.
    with open('enthalpy_of_mixing_reweight.json', 'wb') as file:

        json_results = json.dumps(results, sort_keys=True, indent=2,
                                  separators=(',', ': '), cls=TypedJSONEncoder)

        file.write(json_results.encode('utf-8'))


if __name__ == "__main__":
    main()
