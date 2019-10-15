"""
Units tests for propertyestimator.protocols.miscellaneous
"""
import operator
import random
import tempfile
from functools import reduce

import pytest

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.properties import ParameterGradient, ParameterGradientKey
from propertyestimator.protocols.miscellaneous import AddValues, FilterSubstanceByRole, SubtractValues, MultiplyValue, \
    DivideValue, WeightByMoleFraction
from propertyestimator.substances import Substance
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity


@pytest.mark.parametrize("values", [
    [random.randint(1, 10) for _ in range(10)],
    [random.random() for _ in range(10)],
    [random.random() * unit.kelvin for _ in range(10)],
    [EstimatedQuantity(random.random() * unit.kelvin, random.random() * unit.kelvin, f'{x}') for x in range(10)],
    [ParameterGradient(ParameterGradientKey('a', 'b', 'c'), random.random() * unit.kelvin) for x in range(10)]
])
def test_add_values_protocol(values):

    with tempfile.TemporaryDirectory() as temporary_directory:

        add_quantities = AddValues('add')
        add_quantities.values = values

        result = add_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert add_quantities.result == reduce(operator.add, values)


@pytest.mark.parametrize("values", [
    [random.randint(1, 10) for _ in range(2)],
    [random.random() for _ in range(2)],
    [random.random() * unit.kelvin for _ in range(2)],
    [EstimatedQuantity(random.random() * unit.kelvin, random.random() * unit.kelvin, f'{x}') for x in range(2)],
    [ParameterGradient(ParameterGradientKey('a', 'b', 'c'), random.random() * unit.kelvin) for x in range(2)]
])
def test_subtract_values_protocol(values):

    with tempfile.TemporaryDirectory() as temporary_directory:

        sub_quantities = SubtractValues('sub')
        sub_quantities.value_b = values[1]
        sub_quantities.value_a = values[0]

        result = sub_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert sub_quantities.result == values[1] - values[0]


@pytest.mark.parametrize("value", [
    random.randint(1, 10),
    random.random(),
    random.random() * unit.kelvin,
    EstimatedQuantity(random.random() * unit.kelvin, random.random() * unit.kelvin, 'a'),
    ParameterGradient(ParameterGradientKey('a', 'b', 'c'), random.random() * unit.kelvin)
])
@pytest.mark.parametrize("multiplier", [
    random.randint(1, 10),
    random.random()
])
def test_multiply_values_protocol(value, multiplier):

    with tempfile.TemporaryDirectory() as temporary_directory:

        multiply_quantities = MultiplyValue('multiply')
        multiply_quantities.value = value
        multiply_quantities.multiplier = multiplier

        result = multiply_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert multiply_quantities.result == value * multiplier


@pytest.mark.parametrize("value", [
    random.randint(1, 10),
    random.random(),
    random.random() * unit.kelvin,
    EstimatedQuantity(random.random() * unit.kelvin, random.random() * unit.kelvin, 'a'),
    ParameterGradient(ParameterGradientKey('a', 'b', 'c'), random.random() * unit.kelvin)
])
@pytest.mark.parametrize("divisor", [
    random.randint(1, 10),
    random.random()
])
def test_divide_values_protocol(value, divisor):

    with tempfile.TemporaryDirectory() as temporary_directory:

        divide_quantities = DivideValue('divide')
        divide_quantities.value = value
        divide_quantities.divisor = divisor

        result = divide_quantities.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert divide_quantities.result == value / divisor


@pytest.mark.parametrize("component_smiles", ['C', 'CC', 'CCC'])
@pytest.mark.parametrize("value", [
    random.randint(1, 10),
    random.random(),
    random.random() * unit.kelvin,
    EstimatedQuantity(random.random() * unit.kelvin, random.random() * unit.kelvin, 'a'),
    ParameterGradient(ParameterGradientKey('a', 'b', 'c'), random.random() * unit.kelvin)
])
def test_weight_by_mole_fraction_protocol(component_smiles, value):

    full_substance = Substance.from_components('C', 'CC', 'CCC')
    component = Substance.from_components(component_smiles)

    mole_fraction = next(iter(full_substance.get_amounts(component_smiles))).value

    with tempfile.TemporaryDirectory() as temporary_directory:

        weight_protocol = WeightByMoleFraction('weight')
        weight_protocol.value = value
        weight_protocol.full_substance = full_substance
        weight_protocol.component = component

        result = weight_protocol.execute(temporary_directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert weight_protocol.weighted_value == value * mole_fraction


@pytest.mark.parametrize("filter_role", [Substance.ComponentRole.Solute,
                                         Substance.ComponentRole.Solvent,
                                         Substance.ComponentRole.Ligand,
                                         Substance.ComponentRole.Receptor])
def test_substance_filtering_protocol(filter_role):
    """Tests that the protocol to filter substances by
    role correctly works."""

    def create_substance():
        test_substance = Substance()

        test_substance.add_component(Substance.Component('C', role=Substance.ComponentRole.Solute),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('CC', role=Substance.ComponentRole.Ligand),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('CCC', role=Substance.ComponentRole.Receptor),
                                     Substance.ExactAmount(1))

        test_substance.add_component(Substance.Component('O', role=Substance.ComponentRole.Solvent),
                                     Substance.MoleFraction(1.0))

        return test_substance

    filter_protocol = FilterSubstanceByRole('filter_protocol')
    filter_protocol.input_substance = create_substance()

    filter_protocol.component_role = filter_role
    filter_protocol.execute('', ComputeResources())

    assert len(filter_protocol.filtered_substance.components) == 1
    assert filter_protocol.filtered_substance.components[0].role == filter_role
