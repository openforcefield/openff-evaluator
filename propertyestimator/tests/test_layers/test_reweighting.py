import os
import tempfile
from typing import Iterable

from propertyestimator.datasets import PropertyPhase
from propertyestimator.layers import registered_calculation_schemas
from propertyestimator.layers.reweighting import ReweightingLayer
from propertyestimator.properties import (
    Density,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    DielectricConstant, ExcessMolarVolume)
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import (
    create_dummy_simulation_data,
)


def test_storage_retrieval():
    # Create some dummy properties
    methane = Substance.from_components("C")
    methanol = Substance.from_components("CO")
    mixture = Substance.from_components("C", "CO")
    # Add extra unused data to make sure the wrong data isn't
    # Being retrieved.
    unused_pure = Substance.from_components("CCO")
    unused_mixture = Substance.from_components("CCO", "CO")

    data_to_store = [
        (methane, PropertyPhase.Liquid, 1000),
        (methanol, PropertyPhase.Liquid, 1000),
        (methanol, PropertyPhase.Gas, 1),
        (mixture, PropertyPhase.Liquid, 1000),
        (unused_pure, PropertyPhase.Liquid, 1000),
        (unused_mixture, PropertyPhase.Liquid, 1000),
    ]
    storage_keys = {}

    properties = [
        # Properties with a full system query.
        Density(substance=methanol),
        DielectricConstant(substance=methane),
        # Properties with a multi-component query.
        EnthalpyOfVaporization(substance=methanol),
        # Property with a multi-phase query.
        EnthalpyOfMixing(substance=mixture),
        ExcessMolarVolume(substance=mixture)
    ]
    expected_data_per_property = {
        Density: {"full_system_data": [(methanol, PropertyPhase.Liquid, 1000)]},
        DielectricConstant: {
            "full_system_data": [(methane, PropertyPhase.Liquid, 1000)]
        },
        EnthalpyOfVaporization: {
            "liquid_data": [(methane, PropertyPhase.Liquid, 1000)],
            "gas_data": [(methane, PropertyPhase.Gas, 1)]
        },
        EnthalpyOfMixing: {
            "full_system_data": [(mixture, PropertyPhase.Liquid, 1000)],
            "component_data": [
                [(methane, PropertyPhase.Liquid, 1000)],
                [(methanol, PropertyPhase.Liquid, 1000)],
            ]
        },
        ExcessMolarVolume: {
            "full_system_data": [(mixture, PropertyPhase.Liquid, 1000)],
            "component_data": [
                [(methane, PropertyPhase.Liquid, 1000)],
                [(methanol, PropertyPhase.Liquid, 1000)],
            ]
        },
    }

    with tempfile.TemporaryDirectory() as base_directory:

        # Create a storage backend with some dummy data.
        backend_directory = os.path.join(base_directory, "storage_dir")
        storage_backend = LocalFileStorage(backend_directory)

        for substance, phase, n_mol in data_to_store:

            data_directory = os.path.join(base_directory, substance.identifier)
            data = create_dummy_simulation_data(
                data_directory,
                substance=substance,
                phase=phase,
                number_of_molecules=n_mol,
            )
            storage_key = storage_backend.store_object(data, data_directory)
            storage_keys[(substance, phase, n_mol)] = storage_key

        for physical_property in properties:

            schema = registered_calculation_schemas["ReweightingLayer"][
                physical_property.__class__.__name__
            ]

            if callable(schema):
                schema = schema()

            # noinspection PyTypeChecker
            metadata = ReweightingLayer._get_workflow_metadata(
                physical_property, "", [], storage_backend, schema,
            )

            assert metadata is not None

            expected_data_list = expected_data_per_property[physical_property.__class__]

            for data_key in expected_data_list:

                assert data_key in metadata

                data_list = metadata[data_key]
                assert len(data_list) >= 1

                if not isinstance(data_list[0], list):
                    data_list[0] = [data_list[0]]

                for expected_data in expected_data_list[data_key]:

                    expected_storage_key = storage_keys[expected_data]
