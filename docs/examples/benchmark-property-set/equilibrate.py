import click

from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions

from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, RequestOptions
from openff.evaluator.server.server import EvaluatorServer

from openff.evaluator.forcefield import SmirnoffForceFieldSource

@click.command()
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="dataset.json",
)
@click.option(
    "--n-molecules",
    "-n",
    type=int,
    default=2000,
)
@click.option(
    "--force-field",
    "-f",
    default="openff-2.1.0.offxml",
)
def main(
    dataset_path: str,
    n_molecules: int = 2000,
    force_field: str = "openff-2.1.0.offxml"
):
    # load dataset
    dataset = PhysicalPropertyDataSet.from_json(dataset_path)
    print(f"Loaded {len(dataset.properties)} properties from {dataset_path}")

    options = RequestOptions()
    options.calculation_layers = ["EquilibrationLayer"]
    density_schema = Density.default_equilibration_schema(
        n_molecules=n_molecules,
        absolute_tolerance=100 * unit.kilojoules / unit.mole,
    )

    dhmix_schema = EnthalpyOfMixing.default_equilibration_schema(
        n_molecules=n_molecules,
        absolute_tolerance=100 * unit.kilojoules / unit.mole,
    )

    options.add_schema("EquilibrationLayer", "Density", density_schema)
    options.add_schema("EquilibrationLayer", "EnthalpyOfMixing", dhmix_schema)
    
    force_field_source = SmirnoffForceFieldSource.from_path(
        force_field
    )

    with DaskLocalCluster(
        # uncomment options below to use a GPU to compute (much faster than CPU-only).
        
        number_of_workers=1,
        resources_per_worker=ComputeResources(
            number_of_threads=1,
            number_of_gpus=1,
            preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
        ),
    ) as calculation_backend:
        server = EvaluatorServer(
            calculation_backend=calculation_backend,
            working_directory=".",
            delete_working_files=False
        )
        with server:
            client = EvaluatorClient()

            # we first request the equilibration data
            # this can be copied between different runs to avoid re-running
            # the data is saved in a directory called "stored_data"
        
            request, error = client.request_estimate(
                dataset,
                force_field_source,
                options,
            )

            # block until computation finished
            results, exception = request.results(synchronous=True, polling_interval=30)
            assert exception is None

    print(f"Equilibration complete")
    print(f"# estimated: {len(results.estimated_properties)}")
    print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    print(f"# exceptions: {len(results.exceptions)}")


if __name__ == "__main__":
    main()
