from integration_tests.utils import BackendType, setup_server
from propertyestimator.client import ConnectionOptions, EvaluatorClient, RequestOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import ParameterGradientKey, SmirnoffForceFieldSource
from propertyestimator.utils import setup_timestamp_logging


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_path = "smirnoff99Frosst-1.1.0.offxml"
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Load in the data set containing a single density
    # property.
    with open("pure_data_set.json") as file:
        data_set = PhysicalPropertyDataSet.parse_json(file.read())

    with open("binary_data_set.json") as file:
        data_set.merge(PhysicalPropertyDataSet.parse_json(file.read()))

    # Set up the server object which run the calculations.
    server = setup_server(
        backend_type=BackendType.LocalCPU, max_number_of_workers=1, port=8005
    )

    with server:

        # Request the estimates.
        property_estimator = EvaluatorClient(ConnectionOptions(server_port=8005))

        for calculation_layer in ["SimulationLayer", "ReweightingLayer"]:

            options = RequestOptions()
            options.calculation_layers = [calculation_layer]

            parameter_gradient_keys = [
                ParameterGradientKey(tag='vdW', smirks='[#6X4:1]', attribute='epsilon'),
                ParameterGradientKey(tag='vdW', smirks='[#6X4:1]', attribute='rmin_half')
            ]

            request, _ = property_estimator.request_estimate(
                property_set=data_set,
                force_field_source=force_field_source,
                options=options,
                parameter_gradient_keys=parameter_gradient_keys,
            )

            # Wait for the results.
            results, _ = request.results(True, 5)
            results.json(f"All{calculation_layer}.json", True)


if __name__ == "__main__":
    main()
