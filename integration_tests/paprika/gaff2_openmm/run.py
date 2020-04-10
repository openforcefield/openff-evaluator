#!/usr/bin/env python

from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.backends import ComputeResources
from evaluator.datasets.taproom import TaproomDataSet
from evaluator.forcefield import TLeapForceFieldSource
from evaluator.properties import HostGuestBindingAffinity
from evaluator.utils import setup_timestamp_logging
from evaluator.workflow import Workflow


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_source = TLeapForceFieldSource(
        leap_source="leaprc.gaff2", cutoff=9.0 * unit.angstrom
    )
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
