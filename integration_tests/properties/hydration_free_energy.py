from evaluator.client import ConnectionOptions, EvaluatorClient, RequestOptions
from evaluator.datasets import PhysicalPropertyDataSet
from evaluator.forcefield import SmirnoffForceFieldSource
from evaluator.properties.solvation import SolvationFreeEnergy
from evaluator.utils import setup_timestamp_logging
from integration_tests.utils import BackendType, setup_server


def _get_fixed_lambda_schema():
    """Manually override trailblazing to set the values found in the previous OFF study
     https://github.com/MobleyLab/SMIRNOFF_paper_code/tree/master/FreeSolv

    Returns
    -------
    SimulationSchema
        The simulation schema to use for the hydration free energy
        calculations.
    """

    default_schema = SolvationFreeEnergy.default_simulation_schema()
    workflow_schema = default_schema.workflow_schema

    conditional_group_schema = next(
        x for x in workflow_schema.protocol_schemas if x.id == "conditional_group"
    )
    conditional_group = conditional_group_schema.to_protocol()

    yank_protocol = conditional_group.protocols["run_solvation_yank"]

    yank_protocol.electrostatic_lambdas_1 = [
        1.00,
        0.75,
        0.50,
        0.25,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
    yank_protocol.steric_lambdas_1 = [
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        0.95,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
        0.00,
    ]

    yank_protocol.electrostatic_lambdas_2 = [1.00, 0.75, 0.50, 0.25, 0.00]
    yank_protocol.steric_lambdas_2 = [1.00, 1.00, 1.00, 1.00, 1.00]

    workflow_schema.protocol_schemas.remove(conditional_group_schema)
    workflow_schema.protocol_schemas.append(conditional_group.schema)

    return default_schema


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_path = "smirnoff99Frosst-1.1.0.offxml"
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Create a data set containing three solvation free energies.
    data_set = PhysicalPropertyDataSet.from_json("hydration_data_set.json")
    data_set.json("hydration_data_set.json", format=True)

    # Set up a server object to run the calculations using.
    server = setup_server(
        backend_type=BackendType.LocalGPU, max_number_of_workers=1, port=8002
    )

    with server:

        # Request the estimates.
        property_estimator = EvaluatorClient(ConnectionOptions(server_port=8002))

        options = RequestOptions()
        options.calculation_layers = ["SimulationLayer"]
        options.add_schema(
            "SimulationLayer", "SolvationFreeEnergy", _get_fixed_lambda_schema()
        )

        request, _ = property_estimator.request_estimate(
            property_set=data_set,
            force_field_source=force_field_source,
            options=options,
        )

        # Wait for the results.
        results, _ = request.results(True, 60)

        # Save the result to file.
        results.json(f"results.json", True)


if __name__ == "__main__":
    main()
