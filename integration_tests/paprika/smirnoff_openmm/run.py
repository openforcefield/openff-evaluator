#!/usr/bin/env python

from openforcefield.typing.engines import smirnoff

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.backends import ComputeResources
from openff.evaluator.datasets.taproom import TaproomDataSet
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import HostGuestBindingAffinity
from openff.evaluator.utils import get_data_filename, setup_timestamp_logging
from openff.evaluator.workflow import Workflow


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field = smirnoff.ForceField(
        "smirnoff99Frosst-1.1.0.offxml", get_data_filename("forcefield/tip3p.offxml")
    )

    force_field_source = SmirnoffForceFieldSource.from_object(force_field)
    force_field_source.json("force_field.json")

    # Load in the data set, retaining only a specific host / guest pair.
    host = "acd"
    guest = "bam"

    data_set = TaproomDataSet()
    data_set.filter_by_host_identifiers(host)
    data_set.filter_by_guest_identifiers(guest)

    # Pull out the sole physical property
    binding_affinity = data_set.properties[0]

    # Set up the calculation
    schema = HostGuestBindingAffinity.default_paprika_schema().workflow_schema
    metadata = Workflow.generate_default_metadata(
        binding_affinity, "force_field.json", UNDEFINED
    )

    workflow = Workflow.from_schema(schema, metadata)

    # Run the calculation
    results = workflow.execute(
        root_directory=f"{host}_{guest}",
        compute_resources=ComputeResources(number_of_gpus=1),
    )

    # Save the results
    results.json(f"{host}_{guest}_results.json", format=True)


if __name__ == "__main__":
    main()
