"""
Units tests for the openff.evaluator.client module.
"""

import tempfile

import pytest

from openff.evaluator._tests.utils import create_dummy_property
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, Request, RequestResult
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.forcefield import (
    LigParGenForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.utils.utils import temporarily_change_directory

property_types = [
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
]


def test_default_options():
    """Test creating the default estimation options."""

    data_set = PhysicalPropertyDataSet()
    force_field_source = SmirnoffForceFieldSource.from_path(
        "openff-2.2.1.offxml"
    )

    for property_type in property_types:
        physical_property = create_dummy_property(property_type)
        data_set.add_properties(physical_property)

    options = EvaluatorClient.default_request_options(data_set, force_field_source)
    options.validate()

    assert len(options.calculation_layers) == 2
    assert len(options.calculation_schemas) == len(property_types)
    assert all(
        len(x) == len(options.calculation_layers)
        for x in options.calculation_schemas.values()
    )


@pytest.mark.parametrize(
    "force_field_source, expected_protocol_type",
    [
        (
            SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml"),
            "BuildSmirnoffSystem",
        ),
        (TLeapForceFieldSource(), "BuildTLeapSystem"),
        (LigParGenForceFieldSource(), "BuildLigParGenSystem"),
    ],
)
def test_protocol_replacement(force_field_source, expected_protocol_type):
    data_set = PhysicalPropertyDataSet()

    for property_type in property_types:
        physical_property = create_dummy_property(property_type)
        data_set.add_properties(physical_property)

    options = EvaluatorClient.default_request_options(data_set, force_field_source)
    options_json = options.json(format=True)

    assert options_json.find('BaseBuildSystem"') < 0
    assert options_json.find(expected_protocol_type) >= 0


def test_submission():
    with tempfile.TemporaryDirectory() as directory:
        with temporarily_change_directory(directory):
            with DaskLocalCluster() as calculation_backend:
                # Spin up a server instance.
                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=directory,
                )

                with server:
                    # Connect a client.
                    client = EvaluatorClient()

                    # Submit an empty data set.
                    force_field_path = "openff-2.2.1.offxml"
                    force_field_source = SmirnoffForceFieldSource.from_path(
                        force_field_path
                    )

                    request, error = client.request_estimate(
                        PhysicalPropertyDataSet(), force_field_source
                    )
                    assert error is None
                    assert isinstance(request, Request)

                    result, error = request.results(polling_interval=0.01)
                    assert error is None
                    assert isinstance(result, RequestResult)
