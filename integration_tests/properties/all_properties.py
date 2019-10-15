import json

from propertyestimator import client
from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.properties import ParameterGradientKey
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.tests.utils import setup_server, BackendType
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow import WorkflowOptions


def main(base_output_file_name='results', property_types=None):

    setup_timestamp_logging()

    if property_types is None:
        property_types = [*registered_properties]

    # Load in the force field
    force_field_path = 'smirnoff99Frosst-1.1.0.offxml'
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Load in the data set containing a single density
    # property.
    with open('pure_data_set.json') as file:
        data_set = PhysicalPropertyDataSet.parse_json(file.read())

    with open('binary_data_set.json') as file:
        data_set.merge(PhysicalPropertyDataSet.parse_json(file.read()))

    if property_types is not None:
        data_set.filter_by_property_types(*property_types)

    # Set up the server object which run the calculations.
    setup_server(backend_type=BackendType.LocalGPU, max_number_of_workers=1, port=8005)

    # Request the estimates.
    property_estimator = client.PropertyEstimatorClient(client.ConnectionOptions(server_port=8005))

    options = PropertyEstimatorOptions()
    options.allowed_calculation_layers = ['SimulationLayer']

    options.workflow_options = {}

    for property_type in property_types:

        options.workflow_options[property_type] = {
            'SimulationLayer': WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks),
            'ReweightingLayer': WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks)
        }

    parameter_gradient_keys = [
        ParameterGradientKey(tag='vdW', smirks='[#6X4:1]', attribute='epsilon'),
        ParameterGradientKey(tag='vdW', smirks='[#6X4:1]', attribute='rmin_half')
    ]

    request = property_estimator.request_estimate(property_set=data_set,
                                                  force_field_source=force_field_source,
                                                  options=options,
                                                  parameter_gradient_keys=parameter_gradient_keys)

    # Wait for the results.
    results = request.results(True, 5)

    # Save the result to file.
    with open(f'{base_output_file_name}_simulation.json', 'wb') as file:

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
    with open(f'{base_output_file_name}_reweight.json', 'wb') as file:

        json_results = json.dumps(results, sort_keys=True, indent=2,
                                  separators=(',', ': '), cls=TypedJSONEncoder)

        file.write(json_results.encode('utf-8'))


if __name__ == "__main__":
    main()
