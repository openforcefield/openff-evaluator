from openforcefield.typing.engines.smirnoff import ForceField

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.backends import QueueWorkerResources
from openff.evaluator.backends.dask import DaskLSFBackend
from openff.evaluator.datasets.taproom import TaproomDataSet
from openff.evaluator.forcefield import SmirnoffForceFieldSource, TLeapForceFieldSource
from openff.evaluator.properties import HostGuestBindingAffinity
from openff.evaluator.utils import get_data_filename, setup_timestamp_logging
from openff.evaluator.workflow import Workflow


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field = ForceField(
        "smirnoff99Frosst-1.0.5.offxml", get_data_filename("forcefield/tip3p.offxml")
    )

    force_field_source = SmirnoffForceFieldSource.from_object(force_field)
    force_field_source.json("force-field.json")

    # Load in the data set, retaining only a specific host / guest pair.
    binding_affinity = TaproomDataSet(
        host_codes=["acd"],
        guest_codes=["bam"],
        default_ionic_strength=150 * unit.millimolar,
    ).properties[0]

    # Set up the calculation
    schema = HostGuestBindingAffinity.default_paprika_schema(
        n_solvent_molecules=2000
    ).workflow_schema
    schema.replace_protocol_types(
        {
            "BaseBuildSystem": (
                "BuildSmirnoffSystem"
                if isinstance(force_field_source, SmirnoffForceFieldSource)
                else "BuildTLeapSystem"
                if isinstance(force_field_source, TLeapForceFieldSource)
                else "BaseBuildSystem"
            )
        }
    )

    metadata = Workflow.generate_default_metadata(
        binding_affinity, "force-field.json", UNDEFINED
    )

    workflow = Workflow.from_schema(schema, metadata, "acd_bam")

    # Run the calculation
    with DaskLSFBackend(
        minimum_number_of_workers=1,
        maximum_number_of_workers=50,
        resources_per_worker=QueueWorkerResources(
            number_of_gpus=1,
            preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
            per_thread_memory_limit=5 * unit.gigabyte,
            wallclock_time_limit="05:59",
        ),
        setup_script_commands=[
            "conda activate openff-evaluator-paprika",
            "module load cuda/10.0",
        ],
        queue_name="gpuqueue",
    ) as calculation_backend:
        results = workflow.execute(
            root_directory="workflow", calculation_backend=calculation_backend
        ).result()

    # Save the results
    results.json("results.json", format=True)


if __name__ == "__main__":
    main()
