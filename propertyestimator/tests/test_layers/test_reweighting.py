import os
import tempfile

from propertyestimator.datasets import PropertyPhase
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.layers import registered_calculation_schemas
from propertyestimator.layers.reweighting import ReweightingLayer
from propertyestimator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import create_dummy_simulation_data


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
        ExcessMolarVolume(substance=mixture),
    ]
    expected_data_per_property = {
        Density: {"full_system_data": [(methanol, PropertyPhase.Liquid, 1000)]},
        DielectricConstant: {
            "full_system_data": [(methane, PropertyPhase.Liquid, 1000)]
        },
        EnthalpyOfVaporization: {
            "liquid_data": [(methanol, PropertyPhase.Liquid, 1000)],
            "gas_data": [(methanol, PropertyPhase.Gas, 1)],
        },
        EnthalpyOfMixing: {
            "full_system_data": [(mixture, PropertyPhase.Liquid, 1000)],
            "component_data": [
                [(methane, PropertyPhase.Liquid, 1000)],
                [(methanol, PropertyPhase.Liquid, 1000)],
            ],
        },
        ExcessMolarVolume: {
            "full_system_data": [(mixture, PropertyPhase.Liquid, 1000)],
            "component_data": [
                [(methane, PropertyPhase.Liquid, 1000)],
                [(methanol, PropertyPhase.Liquid, 1000)],
            ],
        },
    }

    force_field = SmirnoffForceFieldSource.from_path("smirnoff99Frosst-1.1.0.offxml")

    with tempfile.TemporaryDirectory() as base_directory:

        # Create a storage backend with some dummy data.
        backend_directory = os.path.join(base_directory, "storage_dir")
        storage_backend = LocalFileStorage(backend_directory)

        force_field_id = storage_backend.store_force_field(force_field)

        for substance, phase, n_mol in data_to_store:

            data_directory = os.path.join(base_directory, substance.identifier)
            data = create_dummy_simulation_data(
                data_directory,
                substance=substance,
                force_field_id=force_field_id,
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

            # noinspection PyProtectedMember
            metadata = ReweightingLayer._get_workflow_metadata(
                base_directory, physical_property, "", [], storage_backend, schema,
            )

            assert metadata is not None

            expected_data_list = expected_data_per_property[physical_property.__class__]

            for data_key in expected_data_list:

                assert data_key in metadata

                stored_metadata = metadata[data_key]
                expected_metadata = expected_data_list[data_key]

                assert len(stored_metadata) == len(expected_metadata)

                if isinstance(stored_metadata[0], list):
                    # Flatten any lists of lists.
                    stored_metadata = [
                        item for sublist in stored_metadata for item in sublist
                    ]
                    expected_metadata = [
                        item for sublist in expected_metadata for item in sublist
                    ]

                metadata_storage_keys = [
                    os.path.basename(x) for x, _, _ in stored_metadata
                ]
                expected_storage_keys = [storage_keys[x] for x in expected_metadata]

                assert sorted(metadata_storage_keys) == sorted(expected_storage_keys)
