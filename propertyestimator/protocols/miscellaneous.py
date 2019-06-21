"""
A collection of protocols for running analysing the results of molecular simulations.
"""
import numpy as np

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

    @protocol_input(list)
    def values(self):
        """The values to add together."""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._values = None
        self._result = None

    def execute(self, directory, available_resources):

        self._result = None

        for value in self._values:

            if self._result is None:

                self._result = value
                continue

            self._result += value

        return self._get_output_dictionary()


@register_calculation_protocol()
class SubtractValues(BaseProtocol):
    """A protocol to subtract one value from another such that:

    `result = value_b - value_a`
    """

    @protocol_input(EstimatedQuantity)
    def value_a(self):
        """`value_a` in the formula `result = value_b - value_a`"""
        pass

    @protocol_input(EstimatedQuantity)
    def value_b(self):
        """`value_b` in the formula  `result = value_b - value_a`"""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

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
class FilterSubstanceByRole(BaseProtocol):
    """A protocol which takes a substance as input, and returns a substance which only
    contains components whose role match a given criteria.
    """

    @protocol_input(Substance)
    def input_substance(self):
        """The substance to filter."""
        pass

    @protocol_input(Substance.ComponentRole)
    def component_role(self):
        """The role to filter substance components against."""
        pass

    @protocol_input(int)
    def expected_components(self):
        """The number of components expected to remain after filtering. An
        exception is raised if this number is not matched. Setting this value
        to -1 will disable this check."""
        pass

    @protocol_output(Substance)
    def filtered_substance(self):
        """The filtered substance."""
        pass

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

            amount = self._input_substance.get_amount(component)

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

            amount = self._input_substance.get_amount(component)

            if isinstance(amount, Substance.MoleFraction):
                amount = Substance.MoleFraction(amount.value * inverse_mole_fraction)

            self._filtered_substance.add_component(component, amount)

        return self._get_output_dictionary()
