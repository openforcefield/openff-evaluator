import numpy
import pandas
import pytest
from openff.units import unit

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.datasets.curation.components.filtering import (
    FilterByPressureSchema,
    FilterByTemperatureSchema,
)
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)
from openff.evaluator.properties import Density
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState


@pytest.fixture(scope="module")
def data_frame() -> pandas.DataFrame:

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        Density(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin,
                pressure=101.325 * unit.kilopascal,
            ),
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("C"),
        ),
        Density(
            thermodynamic_state=ThermodynamicState(
                temperature=305.15 * unit.kelvin,
                pressure=101.325 * unit.kilopascal,
            ),
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("C"),
        ),
        Density(
            thermodynamic_state=ThermodynamicState(
                temperature=298.15 * unit.kelvin,
                pressure=105.325 * unit.kilopascal,
            ),
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("C"),
        ),
    )

    return data_set.to_pandas()


@pytest.fixture(scope="module")
def data_set(data_frame: pandas.DataFrame) -> PhysicalPropertyDataSet:
    return PhysicalPropertyDataSet.from_pandas(data_frame)


def test_workflow_data_frame(data_frame):
    """Test that a simple curation workflow can be applied to a data frame."""

    schema = CurationWorkflowSchema(
        component_schemas=[
            FilterByTemperatureSchema(
                minimum_temperature=290.0, maximum_temperature=300.0
            ),
            FilterByPressureSchema(minimum_pressure=101.3, maximum_pressure=101.4),
        ]
    )

    filtered_frame = CurationWorkflow.apply(data_frame, schema)

    assert isinstance(filtered_frame, pandas.DataFrame)
    assert len(filtered_frame) == 1

    assert numpy.isclose(filtered_frame["Temperature (K)"].values[0], 298.15)
    assert numpy.isclose(filtered_frame["Pressure (kPa)"].values[0], 101.325)


def test_workflow_data_set(data_set):
    """Test that a simple curation workflow can be applied to a data set."""

    schema = CurationWorkflowSchema(
        component_schemas=[
            FilterByTemperatureSchema(
                minimum_temperature=290.0, maximum_temperature=300.0
            ),
            FilterByPressureSchema(minimum_pressure=101.3, maximum_pressure=101.4),
        ]
    )

    filtered_set = CurationWorkflow.apply(data_set, schema)

    assert isinstance(filtered_set, PhysicalPropertyDataSet)
    assert len(filtered_set) == 1

    assert numpy.isclose(
        filtered_set.properties[0].thermodynamic_state.temperature, 298.15 * unit.kelvin
    )
    assert numpy.isclose(
        filtered_set.properties[0].thermodynamic_state.pressure,
        101.325 * unit.kilopascal,
    )
