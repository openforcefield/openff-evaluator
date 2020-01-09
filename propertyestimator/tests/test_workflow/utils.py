from typing import Union

import pint

from propertyestimator.attributes import UNDEFINED
from propertyestimator.layers import registered_calculation_schemas
from propertyestimator.workflow import Workflow
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import Protocol


def create_dummy_metadata(dummy_property, calculation_layer):

    global_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )

    if calculation_layer == "ReweightingLayer":

        schema = registered_calculation_schemas[calculation_layer][
            dummy_property.__class__.__name__
        ]

        if callable(schema):
            schema = schema()

        for key, query in schema.storage_queries.items():

            fake_data = [
                (f"data_path_{index}", f"ff_path_{index}") for index in range(3)
            ]

            if (
                query.substance_query != UNDEFINED
                and query.substance_query.components_only
            ):
                fake_data = [fake_data for _ in dummy_property.substance.components]

            global_metadata[key] = fake_data

    return global_metadata


@workflow_protocol()
class DummyReplicableProtocol(Protocol):

    replicated_value_a = InputAttribute(
        docstring="", type_hint=Union[str, int, float], default_value=UNDEFINED
    )
    replicated_value_b = InputAttribute(
        docstring="", type_hint=Union[str, int, float], default_value=UNDEFINED
    )
    final_value = OutputAttribute(docstring="", type_hint=pint.Measurement)

    def _execute(self, directory, available_resources):
        pass


@workflow_protocol()
class DummyInputOutputProtocol(Protocol):

    input_value = InputAttribute(
        docstring="A dummy input.",
        type_hint=Union[
            str,
            int,
            float,
            pint.Quantity,
            pint.Measurement,
            list,
            tuple,
            dict,
            set,
            frozenset,
        ],
        default_value=UNDEFINED,
    )
    output_value = OutputAttribute(
        docstring="A dummy output.",
        type_hint=Union[
            str,
            int,
            float,
            pint.Quantity,
            pint.Measurement,
            list,
            tuple,
            dict,
            set,
            frozenset,
        ],
    )

    def _execute(self, directory, available_resources):
        self.output_value = self.input_value
