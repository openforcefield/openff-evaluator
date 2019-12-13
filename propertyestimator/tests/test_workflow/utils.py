from typing import Union

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED
from propertyestimator.layers import registered_calculation_schemas
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import Workflow
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


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


@register_calculation_protocol()
class DummyReplicableProtocol(BaseProtocol):

    replicated_value_a = InputAttribute(
        docstring="", type_hint=Union[str, int, float], default_value=UNDEFINED
    )
    replicated_value_b = InputAttribute(
        docstring="", type_hint=Union[str, int, float], default_value=UNDEFINED
    )
    final_value = OutputAttribute(docstring="", type_hint=EstimatedQuantity)


@register_calculation_protocol()
class DummyInputOutputProtocol(BaseProtocol):

    input_value = InputAttribute(
        docstring="A dummy input.",
        type_hint=Union[
            str,
            int,
            float,
            unit.Quantity,
            EstimatedQuantity,
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
            unit.Quantity,
            EstimatedQuantity,
            list,
            tuple,
            dict,
            set,
            frozenset,
        ],
    )

    def execute(self, directory, available_resources):
        self.output_value = self.input_value
        return self._get_output_dictionary()
