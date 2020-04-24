"""
Units tests for evaluator.protocols.miscellaneous
"""
import operator
import random
import tempfile
from functools import reduce

import pytest

from evaluator import unit
from evaluator.backends import ComputeResources
from evaluator.forcefield import ParameterGradient, ParameterGradientKey
from evaluator.protocols.miscellaneous import (
    AddValues,
    AverageFreeEnergies,
    DivideValue,
    FilterSubstanceByRole,
    MultiplyValue,
    SubtractValues,
    WeightByMoleFraction,
)
from evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from evaluator.thermodynamics import ThermodynamicState


@pytest.mark.parametrize(
    "values",
    [
        [random.randint(1, 10) for _ in range(10)],
        [random.random() for _ in range(10)],
        [random.random() * unit.kelvin for _ in range(10)],
        [
            (random.random() * unit.kelvin).plus_minus(random.random() * unit.kelvin)
            for x in range(10)
        ],
        [
            ParameterGradient(
                ParameterGradientKey("a", "b", "c"), random.random() * unit.kelvin
            )
            for _ in range(10)
        ],
    ],
)
def test_add_values_protocol(values):

    with tempfile.TemporaryDirectory() as temporary_directory:

        add_quantities = AddValues("add")
        add_quantities.values = values

        add_quantities.execute(temporary_directory, ComputeResources())
        assert add_quantities.result == reduce(operator.add, values)


@pytest.mark.parametrize(
    "values",
    [
        [random.randint(1, 10) for _ in range(2)],
        [random.random() for _ in range(2)],
        [random.random() * unit.kelvin for _ in range(2)],
        [
            (random.random() * unit.kelvin).plus_minus(random.random() * unit.kelvin)
            for _ in range(2)
        ],
        [
            ParameterGradient(
                ParameterGradientKey("a", "b", "c"), random.random() * unit.kelvin
            )
            for _ in range(2)
        ],
    ],
)
def test_subtract_values_protocol(values):

    with tempfile.TemporaryDirectory() as temporary_directory:

        sub_quantities = SubtractValues("sub")
        sub_quantities.value_b = values[1]
        sub_quantities.value_a = values[0]

        sub_quantities.execute(temporary_directory, ComputeResources())
        assert sub_quantities.result == values[1] - values[0]


@pytest.mark.parametrize(
    "value",
    [
        random.randint(1, 10),
        random.random(),
        random.random() * unit.kelvin,
        (random.random() * unit.kelvin).plus_minus(random.random() * unit.kelvin),
        ParameterGradient(
            ParameterGradientKey("a", "b", "c"), random.random() * unit.kelvin
        ),
    ],
)
@pytest.mark.parametrize("multiplier", [random.randint(1, 10), random.random()])
def test_multiply_values_protocol(value, multiplier):

    with tempfile.TemporaryDirectory() as temporary_directory:

        multiply_quantities = MultiplyValue("multiply")
        multiply_quantities.value = value
        multiply_quantities.multiplier = multiplier
        multiply_quantities.execute(temporary_directory, ComputeResources())
        assert multiply_quantities.result == value * multiplier


@pytest.mark.parametrize(
    "value",
    [
        random.randint(1, 10),
        random.random(),
        random.random() * unit.kelvin,
        (random.random() * unit.kelvin).plus_minus(random.random() * unit.kelvin),
        ParameterGradient(
            ParameterGradientKey("a", "b", "c"), random.random() * unit.kelvin
        ),
    ],
)
@pytest.mark.parametrize("divisor", [random.randint(1, 10), random.random()])
def test_divide_values_protocol(value, divisor):

    with tempfile.TemporaryDirectory() as temporary_directory:

        divide_quantities = DivideValue("divide")
        divide_quantities.value = value
        divide_quantities.divisor = divisor
        divide_quantities.execute(temporary_directory, ComputeResources())
        assert divide_quantities.result == value / divisor


@pytest.mark.parametrize("component_smiles", ["C", "CC", "CCC"])
@pytest.mark.parametrize(
    "value",
    [
        random.randint(1, 10),
        random.random(),
        random.random() * unit.kelvin,
        (random.random() * unit.kelvin).plus_minus(random.random() * unit.kelvin),
        ParameterGradient(
            ParameterGradientKey("a", "b", "c"), random.random() * unit.kelvin
        ),
    ],
)
def test_weight_by_mole_fraction_protocol(component_smiles, value):

    full_substance = Substance.from_components("C", "CC", "CCC")
    component = Substance.from_components(component_smiles)

    mole_fraction = next(
        iter(full_substance.get_amounts(component.components[0].identifier))
    ).value

    with tempfile.TemporaryDirectory() as temporary_directory:

        weight_protocol = WeightByMoleFraction("weight")
        weight_protocol.value = value
        weight_protocol.full_substance = full_substance
        weight_protocol.component = component
        weight_protocol.execute(temporary_directory, ComputeResources())
        assert weight_protocol.weighted_value == value * mole_fraction


@pytest.mark.parametrize(
    "filter_role",
    [
        Component.Role.Solute,
        Component.Role.Solvent,
        Component.Role.Ligand,
        Component.Role.Receptor,
    ],
)
def test_substance_filtering_protocol(filter_role):
    """Tests that the protocol to filter substances by
    role correctly works."""

    def create_substance():
        test_substance = Substance()

        test_substance.add_component(
            Component("C", role=Component.Role.Solute), ExactAmount(1),
        )

        test_substance.add_component(
            Component("CC", role=Component.Role.Ligand), ExactAmount(1),
        )

        test_substance.add_component(
            Component("CCC", role=Component.Role.Receptor), ExactAmount(1),
        )

        test_substance.add_component(
            Component("O", role=Component.Role.Solvent), MoleFraction(1.0),
        )

        return test_substance

    filter_protocol = FilterSubstanceByRole("filter_protocol")
    filter_protocol.input_substance = create_substance()

    filter_protocol.component_roles = [filter_role]
    filter_protocol.execute("", ComputeResources())

    assert len(filter_protocol.filtered_substance.components) == 1
    assert filter_protocol.filtered_substance.components[0].role == filter_role


def test_average_free_energies_protocol():
    """Tests adding together two free energies."""

    compute_resources = ComputeResources(number_of_threads=1)

    delta_g_one = (-10.0 * unit.kilocalorie / unit.mole).plus_minus(
        1.0 * unit.kilocalorie / unit.mole
    )
    delta_g_two = (-20.0 * unit.kilocalorie / unit.mole).plus_minus(
        2.0 * unit.kilocalorie / unit.mole
    )

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1 * unit.atmosphere)

    sum_protocol = AverageFreeEnergies("average_free_energies")

    sum_protocol.values = [delta_g_one, delta_g_two]
    sum_protocol.thermodynamic_state = thermodynamic_state

    sum_protocol.execute("", compute_resources)

    result_value = sum_protocol.result.value.to(unit.kilocalorie / unit.mole)
    result_uncertainty = sum_protocol.result.error.to(unit.kilocalorie / unit.mole)

    assert isinstance(sum_protocol.result, unit.Measurement)
    assert result_value.magnitude == pytest.approx(-20.0, abs=0.2)
    assert result_uncertainty.magnitude == pytest.approx(2.0, abs=0.2)
