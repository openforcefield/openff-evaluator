"""
A collection of miscellaneous protocols, mostly aimed at performing simple
math operations.
"""
import typing

import numpy as np

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED
from propertyestimator.forcefield import ParameterGradient
from propertyestimator.substances import Component, MoleFraction, Substance
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import Protocol


@workflow_protocol()
class AddValues(Protocol):
    """A protocol to add together a list of values.

    Notes
    -----
    The `values` input must either be a list of unit.Quantity, a ProtocolPath to a list
    of unit.Quantity, or a list of ProtocolPath which each point to a unit.Quantity.
    """

    values = InputAttribute(
        docstring="The values to add together.", type_hint=list, default_value=UNDEFINED
    )

    result = OutputAttribute(
        docstring="The sum of the values.",
        type_hint=typing.Union[
            int, float, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
    )

    def execute(self, directory, available_resources):

        if len(self.values) < 1:
            raise ValueError("There were no values to add together")

        if not all(isinstance(x, type(self.values[0])) for x in self.values):

            types = " ".join(map(str, self.values))

            raise ValueError(
                f"All of the values to add together must be the same type ({types})."
            )

        self.result = self.values[0]

        for value in self.values[1:]:
            self.result += value

        return self._get_output_dictionary()


@workflow_protocol()
class SubtractValues(Protocol):
    """A protocol to subtract one value from another such that:

    `result = value_b - value_a`
    """

    value_a = InputAttribute(
        docstring="`value_a` in the formula `result` = `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, unit.Quantity, EstimatedQuantity, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    value_b = InputAttribute(
        docstring="`value_b` in the formula `result` = `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, unit.Quantity, EstimatedQuantity, ParameterGradient
        ],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The results of `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
    )

    def execute(self, directory, available_resources):

        self.result = self.value_b - self.value_a
        return self._get_output_dictionary()


@workflow_protocol()
class MultiplyValue(Protocol):
    """A protocol which multiplies a value by a specified scalar
    """

    value = InputAttribute(
        docstring="The value to multiply.",
        type_hint=typing.Union[
            int, float, unit.Quantity, EstimatedQuantity, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    multiplier = InputAttribute(
        docstring="The scalar to multiply by.",
        type_hint=typing.Union[int, float, unit.Quantity],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The result of the multiplication.",
        type_hint=typing.Union[
            int, float, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
    )

    def execute(self, directory, available_resources):

        if isinstance(self.value, EstimatedQuantity):

            self.result = EstimatedQuantity(
                self.value.value * self.multiplier,
                self.value.uncertainty * self.multiplier,
                *self.value.sources,
            )

        else:

            self.result = self.value * self.multiplier

        return self._get_output_dictionary()


@workflow_protocol()
class DivideValue(Protocol):
    """A protocol which divides a value by a specified scalar
    """

    value = InputAttribute(
        docstring="The value to divide.",
        type_hint=typing.Union[
            int, float, unit.Quantity, EstimatedQuantity, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    divisor = InputAttribute(
        docstring="The scalar to divide by.",
        type_hint=typing.Union[int, float, unit.Quantity],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The result of the division.",
        type_hint=typing.Union[
            int, float, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
    )

    def execute(self, directory, available_resources):

        self.result = self.value / self.divisor
        return self._get_output_dictionary()


@workflow_protocol()
class WeightByMoleFraction(Protocol):
    """Multiplies a value by the mole fraction of a component
    in a `Substance`.
    """

    value = InputAttribute(
        docstring="The value to be weighted.",
        type_hint=typing.Union[
            float, int, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
        default_value=UNDEFINED,
    )

    component = InputAttribute(
        docstring="The component whose mole fraction to weight by.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    full_substance = InputAttribute(
        docstring="The full substance which describes the mole fraction of the component.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    weighted_value = OutputAttribute(
        "The value weighted by the `component`s mole fraction as determined from the "
        "`full_substance`.",
        type_hint=typing.Union[
            float, int, EstimatedQuantity, unit.Quantity, ParameterGradient
        ],
    )

    def _weight_values(self, mole_fraction):
        """Weights a value by a components mole fraction.

        Parameters
        ----------
        mole_fraction: float
            The mole fraction to weight by.

        Returns
        -------
        float, int, EstimatedQuantity, unit.Quantity, ParameterGradient
            The weighted value.
        """
        return self.value * mole_fraction

    def execute(self, directory, available_resources):

        assert len(self.component.components) == 1

        main_component = self.component.components[0]
        amounts = self.full_substance.get_amounts(main_component)

        if len(amounts) != 1:

            raise ValueError(
                f"More than one type of amount was defined for component "
                f"{main_component}. Only a single mole fraction must be defined.",
            )

        amount = next(iter(amounts))

        if not isinstance(amount, MoleFraction):

            raise ValueError(
                f"The component {main_component} was given as an exact amount, and "
                f"not a mole fraction"
            )

        self.weighted_value = self._weight_values(amount.value)
        return self._get_output_dictionary()


@workflow_protocol()
class FilterSubstanceByRole(Protocol):
    """A protocol which takes a substance as input, and returns a substance which only
    contains components whose role match a given criteria.
    """

    input_substance = InputAttribute(
        docstring="The substance to filter.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    component_role = InputAttribute(
        docstring="The role to filter substance components against.",
        type_hint=Component.Role,
        default_value=UNDEFINED,
    )

    expected_components = InputAttribute(
        docstring="The number of components expected to remain after filtering. "
        "An exception is raised if this number is not matched.",
        type_hint=int,
        default_value=UNDEFINED,
        optional=True,
    )

    filtered_substance = OutputAttribute(
        docstring="The filtered substance.", type_hint=Substance
    )

    def execute(self, directory, available_resources):

        filtered_components = []
        total_mole_fraction = 0.0

        for component in self.input_substance.components:

            if component.role != self.component_role:
                continue

            filtered_components.append(component)

            amounts = self.input_substance.get_amounts(component)

            for amount in amounts:

                if not isinstance(amount, MoleFraction):
                    continue

                total_mole_fraction += amount.value

        if self.expected_components != UNDEFINED and self.expected_components != len(
            filtered_components
        ):

            raise ValueError(
                f"The filtered substance does not contain the expected number of "
                f"components ({self.expected_components}) - {filtered_components}",
            )

        inverse_mole_fraction = (
            1.0 if np.isclose(total_mole_fraction, 0.0) else 1.0 / total_mole_fraction
        )

        self.filtered_substance = Substance()

        for component in filtered_components:

            amounts = self.input_substance.get_amounts(component)

            for amount in amounts:

                if isinstance(amount, MoleFraction):
                    amount = MoleFraction(amount.value * inverse_mole_fraction)

                self.filtered_substance.add_component(component, amount)

        return self._get_output_dictionary()
