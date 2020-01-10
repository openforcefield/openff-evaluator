"""
Units tests for the propertyestimator.client module.
"""
import tempfile

import pytest

from propertyestimator.backends import DaskLocalCluster
from propertyestimator.client import EvaluatorClient, Request, RequestResult
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import (
    LigParGenForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from propertyestimator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from propertyestimator.server import EvaluatorServer
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.utils.utils import temporarily_change_directory

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
        "smirnoff99Frosst-1.1.0.offxml"
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
            SmirnoffForceFieldSource.from_path("smirnoff99Frosst-1.1.0.offxml"),
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
                    force_field_path = "smirnoff99Frosst-1.1.0.offxml"
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
