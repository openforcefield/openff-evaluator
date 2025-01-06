import tempfile

import pytest
from openff.units import unit

from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, Request, RequestOptions
from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.forcefield import (
    LigParGenForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import Batch, EvaluatorServer
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState


def _modify_schema_dummy(schema):
    workflow_schema = schema.workflow_schema
    workflow_schema.replace_protocol_types({"BaseBuildSystem": "BuildSmirnoffSystem"})
    for schema in workflow_schema.protocol_schemas:
        # shorten protocols
        if "conditional" in schema.id:
            for protocol_name, protocol in schema.protocol_schemas.items():
                if "simulation" in protocol_name:
                    protocol.inputs[".steps_per_iteration"] = 10
                    protocol.inputs[".output_frequency"] = 10
                    protocol.inputs[".checkpoint_frequency"] = 10


class TestPreequilibratedLayer:

    @pytest.fixture
    def dataset(self):
        dataset = PhysicalPropertyDataSet()
        thermodynamic_state = ThermodynamicState(
            temperature=298.15 * unit.kelvin,
            pressure=101.325 * unit.kilopascal,
        )
        dataset.add_properties(
            Density(
                thermodynamic_state=thermodynamic_state,
                phase=PropertyPhase.Liquid,
                value=1.0 * Density.default_unit(),
                uncertainty=1.0 * Density.default_unit(),
                source=MeasurementSource(doi=" "),
                substance=Substance.from_components("CCCO"),
            ),
            EnthalpyOfMixing(
                thermodynamic_state=thermodynamic_state,
                phase=PropertyPhase.Liquid,
                value=1.0 * EnthalpyOfMixing.default_unit(),
                uncertainty=1.0 * EnthalpyOfMixing.default_unit(),
                source=MeasurementSource(doi=" "),
                substance=Substance.from_components("CCCO", "O"),
            ),
        )

        return dataset

    def test_full_run(self, dataset, tmpdir):
        equilibration_options = RequestOptions()
        equilibration_options.calculation_layers = ["EquilibrationLayer"]
        density_equilibration_schema = Density.default_equilibration_schema(
            n_molecules=256,
        )
        _modify_schema_dummy(density_equilibration_schema)

        dhmix_equilibration_schema = EnthalpyOfMixing.default_equilibration_schema(
            n_molecules=256,
        )
        _modify_schema_dummy(dhmix_equilibration_schema)

        equilibration_options.add_schema(
            "EquilibrationLayer",
            "Density",
            density_equilibration_schema,
        )
        equilibration_options.add_schema(
            "EquilibrationLayer",
            "EnthalpyOfMixing",
            dhmix_equilibration_schema,
        )

        preequilibrated_simulation_options = RequestOptions()
        preequilibrated_simulation_options.calculation_layers = [
            "PreequilibratedSimulationLayer"
        ]
        density_preequilibration_schema = (
            Density.default_preequilibrated_simulation_schema(
                n_molecules=256,
            )
        )
        _modify_schema_dummy(density_preequilibration_schema)
        dhmix_preequilibration_schema = (
            EnthalpyOfMixing.default_preequilibrated_simulation_schema(
                n_molecules=256,
            )
        )
        _modify_schema_dummy(dhmix_preequilibration_schema)
        preequilibrated_simulation_options.add_schema(
            "PreequilibratedSimulationLayer",
            "Density",
            density_preequilibration_schema,
        )
        preequilibrated_simulation_options.add_schema(
            "PreequilibratedSimulationLayer",
            "EnthalpyOfMixing",
            dhmix_preequilibration_schema,
        )

        with tmpdir.as_cwd():

            # start server
            with DaskLocalCluster() as calculation_backend:
                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=".",
                )
                with server:
                    client = EvaluatorClient()

                    force_field_path = "openff-2.1.0.offxml"
                    force_field_source = SmirnoffForceFieldSource.from_path(
                        force_field_path
                    )

                    # test equilibration
                    request, error = client.request_estimate(
                        dataset,
                        force_field_source,
                        equilibration_options,
                    )
                    assert error is None
                    results, exception = request.results(
                        synchronous=True, polling_interval=30
                    )

                    assert exception is None

                    request, error = client.request_estimate(
                        dataset,
                        force_field_source,
                        preequilibrated_simulation_options,
                    )
                    assert error is None
                    results, exception = request.results(
                        synchronous=True, polling_interval=30
                    )
                    assert exception is None

        assert len(results.estimated_properties) == 2
