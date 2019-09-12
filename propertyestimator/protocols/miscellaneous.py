"""
A collection of miscellaneous protocols, mostly aimed at performing simple
math operations.
"""
import numpy as np
import typing

from propertyestimator import unit
from propertyestimator.properties import ParameterGradient
from propertyestimator.substances import Substance
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class AddValues(BaseProtocol):
    """A protocol to add together a list of values.

    Notes
    -----
    The `values` input must either be a list of unit.Quantity, a ProtocolPath to a list
    of unit.Quantity, or a list of ProtocolPath which each point to a unit.Quantity.
    """

    values = protocol_input(docstring='The values to add together.',
                            type_hint=list,
                            default_value=protocol_input.UNDEFINED)

    result = protocol_output(docstring='The sum of the values.',
                               type_hint=typing.Union[int, float, EstimatedQuantity, unit.Quantity, ParameterGradient])

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._values = None
        self._result = None

    def execute(self, directory, available_resources):

        if len(self._values) < 1:
            return PropertyEstimatorException(directory, 'There were no gradients to add together')

        if not all(isinstance(x, type(self._values[0])) for x in self._values):

            return PropertyEstimatorException(directory, f'All values to add together must be '
                                                         f'the same type ({" ".join(map(str, self._values))}).')

        self._result = self._values[0]

        for value in self._values[1:]:
            self._result += value

        return self._get_output_dictionary()


@register_calculation_protocol()
class SubtractValues(BaseProtocol):
    """A protocol to subtract one value from another such that:

    `result = value_b - value_a`
    """

    value_a = protocol_input(docstring='`value_a` in the formula `result` = `value_b` - `value_a`.',
                             type_hint=typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient],
                             default_value=protocol_input.UNDEFINED)

    value_b = protocol_input(docstring='`value_b` in the formula `result` = `value_b` - `value_a`.',
                             type_hint=typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient],
                             default_value=protocol_input.UNDEFINED)

    result = protocol_output(docstring='The results of `value_b` - `value_a`.',
                             type_hint=typing.Union[int, float, EstimatedQuantity, unit.Quantity, ParameterGradient])

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._value_a = None
        self._value_b = None

        self._result = None

    def execute(self, directory, available_resources):

        self._result = self._value_b - self._value_a
        return self._get_output_dictionary()


@register_calculation_protocol()
class MultiplyValue(BaseProtocol):
    """A protocol which multiplies a value by a specified scalar
    """

    value = protocol_input(docstring='The value to multiply.',
                           type_hint=typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient],
                           default_value=protocol_input.UNDEFINED)

    multiplier = protocol_input(docstring='The scalar to multiply by.',
                                type_hint=typing.Union[int, float, unit.Quantity],
                                default_value=protocol_input.UNDEFINED)

    result = protocol_output(docstring='The result of the multiplication.',
                             type_hint=typing.Union[int, float, EstimatedQuantity, unit.Quantity, ParameterGradient])

    def __init__(self, protocol_id):
        """Constructs a new MultiplyValue object."""
        super().__init__(protocol_id)

        self._value = None
        self._multiplier = None

        self._result = None

    def execute(self, directory, available_resources):

        if isinstance(self._value, EstimatedQuantity):

            self._result = EstimatedQuantity(self._value.value * self._multiplier,
                                             self._value.uncertainty * self._multiplier,
                                             *self._value.sources)

        else:

            self._result = self._value * self._multiplier

        return self._get_output_dictionary()


@register_calculation_protocol()
class DivideValue(BaseProtocol):
    """A protocol which divides a value by a specified scalar
    """

    value = protocol_input(docstring='The value to divide.',
                           type_hint=typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient],
                           default_value=protocol_input.UNDEFINED)

    divisor = protocol_input(docstring='The scalar to divide by.',
                             type_hint=typing.Union[int, float, unit.Quantity],
                             default_value=protocol_input.UNDEFINED)

    result = protocol_output(docstring='The result of the division.',
                             type_hint=typing.Union[int, float, EstimatedQuantity, unit.Quantity, ParameterGradient])

    def __init__(self, protocol_id):
        """Constructs a new DivideValue object."""
        super().__init__(protocol_id)

        self._value = None
        self._divisor = None

        self._result = None

    def execute(self, directory, available_resources):

        self._result = self._value / self._divisor
        return self._get_output_dictionary()


@register_calculation_protocol()
class BaseWeightByMoleFraction(BaseProtocol):
    """Multiplies a value by the mole fraction of a component
    in a mixture substance.
    """

    component = protocol_input(docstring='The component (e.g water) whose mole fraction to weight by.',
                               type_hint=Substance,
                               default_value=protocol_input.UNDEFINED)

    full_substance = protocol_input(docstring='The full substance which describes the mole fraction of the component.',
                                    type_hint=Substance,
                                    default_value=protocol_input.UNDEFINED)

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._value = None
        self._component = None
        self._full_substance = None

        self._weighted_value = None

    def _weight_values(self, mole_fraction):
        """Weights a value by a components mole fraction.

        Parameters
        ----------
        mole_fraction: float
            The mole fraction to weight by.

        Returns
        -------
        Any
            The weighted value.
        """
        raise NotImplementedError()

    def execute(self, directory, available_resources):

        assert len(self._component.components) == 1

        main_component = self._component.components[0]
        amounts = self._full_substance.get_amounts(main_component)

        if len(amounts) != 1:

            return PropertyEstimatorException(directory=directory,
                                              message=f'More than one type of amount was defined for component '
                                                      f'{main_component}. Only a single mole fraction must be '
                                                      f'defined.')

        amount = next(iter(amounts))

        if not isinstance(amount, Substance.MoleFraction):

            return PropertyEstimatorException(directory=directory,
                                              message=f'The component {main_component} was given as an '
                                                      f'exact amount, and not a mole fraction')

        self._weighted_value = self._weight_values(amount.value)
        return self._get_output_dictionary()


@register_calculation_protocol()
class WeightByMoleFraction(BaseWeightByMoleFraction):
    """Multiplies a value by the mole fraction of a component
    in a `Substance`.
    """

    value = protocol_input(docstring='The value to be weighted.',
                           type_hint=typing.Union[float, int, EstimatedQuantity, unit.Quantity, ParameterGradient],
                           default_value=protocol_input.UNDEFINED)

    weighted_value = protocol_output('The value weighted by the `component`s mole fraction as determined from the '
                                     '`full_substance`.',
                                     type_hint=typing.Union[float, int, EstimatedQuantity, unit.Quantity,
                                                            ParameterGradient])

    def _weight_values(self, mole_fraction):
        """
        Returns
        -------
        float, int, EstimatedQuantity, unit.Quantity, ParameterGradient
            The weighted value.
        """
        return self._value * mole_fraction


@register_calculation_protocol()
class FilterSubstanceByRole(BaseProtocol):
    """A protocol which takes a substance as input, and returns a substance which only
    contains components whose role match a given criteria.
    """

    input_substance = protocol_input(docstring='The substance to filter.',
                                     type_hint=Substance,
                                     default_value=protocol_input.UNDEFINED)

    component_role = protocol_input(docstring='The role to filter substance components against.',
                                    type_hint=Substance.ComponentRole,
                                    default_value=protocol_input.UNDEFINED)

    expected_components = protocol_input(docstring='The number of components expected to remain after filtering. '
                                                   'An exception is raised if this number is not matched.',
                                         type_hint=int,
                                         default_value=protocol_input.UNDEFINED,
                                         optional=True)

    filtered_substance = protocol_output(docstring='The filtered substance.',
                                         type_hint=Substance)

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._input_substance = None
        self._component_role = None

        self._expected_components = -1

        self._filtered_substance = None

    def execute(self, directory, available_resources):

        filtered_components = []
        total_mole_fraction = 0.0

        for component in self._input_substance.components:

            if component.role != self._component_role:
                continue

            filtered_components.append(component)

            amounts = self._input_substance.get_amounts(component)

            for amount in amounts:

                if not isinstance(amount, Substance.MoleFraction):
                    continue

                total_mole_fraction += amount.value

        if 0 <= self._expected_components != len(filtered_components):

            return PropertyEstimatorException(directory=directory,
                                              message=f'The filtered substance does not contain the expected '
                                                      f'number of components ({self._expected_components}) - '
                                                      f'{filtered_components}')

        inverse_mole_fraction = 1.0 if np.isclose(total_mole_fraction, 0.0) else 1.0 / total_mole_fraction

        self._filtered_substance = Substance()

        for component in filtered_components:

            amounts = self._input_substance.get_amounts(component)

            for amount in amounts:

                if isinstance(amount, Substance.MoleFraction):
                    amount = Substance.MoleFraction(amount.value * inverse_mole_fraction)

                self._filtered_substance.add_component(component, amount)

        return self._get_output_dictionary()
