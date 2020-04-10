"""
A collection of miscellaneous protocols, mostly aimed at performing simple
math operations.
"""
import typing

import numpy as np
import pint

from evaluator.attributes import UNDEFINED
from evaluator.forcefield import ParameterGradient
from evaluator.substances import Component, MoleFraction, Substance
from evaluator.workflow.attributes import InputAttribute, OutputAttribute
from evaluator.workflow.plugins import workflow_protocol
from evaluator.workflow.protocols import Protocol


@workflow_protocol()
class AddValues(Protocol):
    """A protocol to add together a list of values.

    Notes
    -----
    The `values` input must either be a list of pint.Quantity, a ProtocolPath to a list
    of pint.Quantity, or a list of ProtocolPath which each point to a pint.Quantity.
    """

    values = InputAttribute(
        docstring="The values to add together.", type_hint=list, default_value=UNDEFINED
    )

    result = OutputAttribute(
        docstring="The sum of the values.",
        type_hint=typing.Union[
            int, float, pint.Measurement, pint.Quantity, ParameterGradient
        ],
    )

    def _execute(self, directory, available_resources):

        if len(self.values) < 1:
            raise ValueError("There were no values to add together")

        self.result = self.values[0]

        for value in self.values[1:]:
            self.result += value


@workflow_protocol()
class SubtractValues(Protocol):
    """A protocol to subtract one value from another such that:

    `result = value_b - value_a`
    """

    value_a = InputAttribute(
        docstring="`value_a` in the formula `result` = `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, pint.Quantity, pint.Measurement, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    value_b = InputAttribute(
        docstring="`value_b` in the formula `result` = `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, pint.Quantity, pint.Measurement, ParameterGradient
        ],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The results of `value_b` - `value_a`.",
        type_hint=typing.Union[
            int, float, pint.Measurement, pint.Quantity, ParameterGradient
        ],
    )

    def _execute(self, directory, available_resources):
        self.result = self.value_b - self.value_a


@workflow_protocol()
class MultiplyValue(Protocol):
    """A protocol which multiplies a value by a specified scalar
    """

    value = InputAttribute(
        docstring="The value to multiply.",
        type_hint=typing.Union[
            int, float, pint.Quantity, pint.Measurement, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    multiplier = InputAttribute(
        docstring="The scalar to multiply by.",
        type_hint=typing.Union[int, float, pint.Quantity],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The result of the multiplication.",
        type_hint=typing.Union[
            int, float, pint.Measurement, pint.Quantity, ParameterGradient
        ],
    )

    def _execute(self, directory, available_resources):
        self.result = self.value * self.multiplier


@workflow_protocol()
class DivideValue(Protocol):
    """A protocol which divides a value by a specified scalar
    """

    value = InputAttribute(
        docstring="The value to divide.",
        type_hint=typing.Union[
            int, float, pint.Quantity, pint.Measurement, ParameterGradient
        ],
        default_value=UNDEFINED,
    )
    divisor = InputAttribute(
        docstring="The scalar to divide by.",
        type_hint=typing.Union[int, float, pint.Quantity],
        default_value=UNDEFINED,
    )

    result = OutputAttribute(
        docstring="The result of the division.",
        type_hint=typing.Union[
            int, float, pint.Measurement, pint.Quantity, ParameterGradient
        ],
    )

    def _execute(self, directory, available_resources):
        self.result = self.value / self.divisor


@workflow_protocol()
class WeightByMoleFraction(Protocol):
    """Multiplies a value by the mole fraction of a component
    in a `Substance`.
    """

    value = InputAttribute(
        docstring="The value to be weighted.",
        type_hint=typing.Union[
            float, int, pint.Measurement, pint.Quantity, ParameterGradient
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
            float, int, pint.Measurement, pint.Quantity, ParameterGradient
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
        float, int, pint.Measurement, pint.Quantity, ParameterGradient
            The weighted value.
        """
        return self.value * mole_fraction

    def _execute(self, directory, available_resources):

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

    def _execute(self, directory, available_resources):

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


@register_calculation_protocol()
class AddBindingFreeEnergies(BaseProtocol):
    """A protocol to add together a list of binding free energies.

    Notes
    -----
    The `values` input must either be a list of EstimatedQuantity, a ProtocolPath to a list
    of EstimatedQuantity, or a list of ProtocolPath which each point to a EstimatedQuantity.
    """

    values = protocol_input(
        docstring='The values to add together.',
        type_hint=list,
        default_value=UNDEFINED
    )
    thermodynamic_state = protocol_input(
        docstring='The thermodynamic state at which the free energies were measured.',
        type_hint=ThermodynamicState,
        default_value=UNDEFINED
    )

    bootstrap_cycles = protocol_input(
        docstring='The number of bootstrap cycles to perform when estimating '
                  'the uncertainty in the combined free energies.',
        type_hint=int,
        default_value=2000
    )

    result = protocol_output(
        docstring='The sum of the values.',
        type_hint=EstimatedQuantity
    )
    confidence_intervals = protocol_output(
        docstring='The confidence intervals on the summed free energy.',
        type_hint=unit.Quantity
    )

    def execute(self, directory, available_resources):

        mean, uncertainty, confidence_intervals = self.bootstrap()

        self.result = EstimatedQuantity(mean,
                                        uncertainty,
                                        self._id)

        self.confidence_intervals = confidence_intervals

        return self._get_output_dictionary()

    def bootstrap(self):
        """

        Returns
        -------
        unit.Quantity
            The summed free energies.
        unit.Quantity
            The uncertainty in the summed free energies
        unit.Quantity
            A unit wrapped list of the confidence intervals.
        """

        default_unit = unit.kilocalorie / unit.mole

        boltzmann_factor = self.thermodynamic_state.temperature * unit.molar_gas_constant
        boltzmann_factor.ito(default_unit)

        beta = 1.0 / boltzmann_factor

        cycle_result = np.empty(self.bootstrap_cycles)

        for cycle_index, cycle in enumerate(range(self.bootstrap_cycles)):

            cycle_values = np.empty(len(self.values))

            for value_index, value in enumerate(self.values):

                mean = value.value.to(default_unit).magnitude
                sem = value.uncertainty.to(default_unit).magnitude

                sampled_value = np.random.normal(mean, sem) * default_unit
                cycle_values[value_index] = (-beta * sampled_value).to(unit.dimensionless).magnitude

            # ΔG° = -RT × Log[ Σ_{n} exp(-βΔG°_{n}) ]
            cycle_result[cycle_index] = np.log(np.sum(np.exp(cycle_values)))

        mean = np.mean(-boltzmann_factor * cycle_result)
        sem = np.std(-boltzmann_factor * cycle_result)

        confidence_intervals = np.empty(2)
        sorted_statistics = np.sort(cycle_result)
        confidence_intervals[0] = sorted_statistics[int(0.025 * self.bootstrap_cycles)]
        confidence_intervals[1] = sorted_statistics[int(0.975 * self.bootstrap_cycles)]

        confidence_intervals = -boltzmann_factor * confidence_intervals

        return mean, sem, confidence_intervals


@register_calculation_protocol()
class AddBindingEnthalpies(BaseProtocol):
    """A protocol to add together a list of binding free enthalpies.

    Notes
    -----
    The `values` input must either be a list of EstimatedQuantity, a ProtocolPath to a list
    of EstimatedQuantity, or a list of ProtocolPath which each point to a EstimatedQuantity.

    With multiple binding orientations, the binding enthalpy of each orientation is weighted its respective
    binding free energy, and therefore this class must accept both binding enthalpies and binding free energies.

    For more information, see:
    Computational Calorimetry: High-Precision Calculation of Host–Guest Binding Thermodynamics
    Niel M. Henriksen, Andrew T. Fenley, Michael K. Gilson
    Journal of Chemical Theory and Computation (2015-08-26) https://doi.org/f7q3mj
    DOI: 10.1021/acs.jctc.5b00405 · PMID: 26523125 · PMCID: PMC4614838
    """

    enthalpy_free_energy_tuple = protocol_input(
        docstring='The enthalpies to add together, passed as a tuple '
                  'with their respective binding free energies.',
        type_hint=list,
        default_value=UNDEFINED
    )
    thermodynamic_state = protocol_input(
        docstring='The thermodynamic state at which the free energies were measured.',
        type_hint=ThermodynamicState,
        default_value=UNDEFINED
    )

    bootstrap_cycles = protocol_input(
        docstring='The number of bootstrap cycles to perform when estimating '
                  'the uncertainty in the combined free energies.',
        type_hint=int,
        default_value=2000
    )

    result = protocol_output(
        docstring='The sum of the enthalpies.',
        type_hint=EstimatedQuantity
    )
    confidence_intervals = protocol_output(
        docstring='The confidence intervals on the summed enthalpy.',
        type_hint=unit.Quantity
    )

    def execute(self, directory, available_resources):

        mean, uncertainty, confidence_intervals = self.bootstrap()

        self.result = EstimatedQuantity(mean,
                                        uncertainty,
                                        self._id)

        self.confidence_intervals = confidence_intervals

        return self._get_output_dictionary()

    def bootstrap(self):
        """

        Returns
        -------
        unit.Quantity
            The summed enthalpies.
        unit.Quantity
            The uncertainty in the summed enthalpies
        unit.Quantity
            A unit wrapped list of the confidence intervals.
        """

        default_unit = unit.kilocalorie / unit.mole

        boltzmann_factor = self.thermodynamic_state.temperature * unit.molar_gas_constant
        boltzmann_factor.ito(default_unit)

        beta = 1.0 / boltzmann_factor

        cycle_result = np.empty(self.bootstrap_cycles)

        for cycle_index, cycle in enumerate(range(self.bootstrap_cycles)):

            cycle_values = np.empty((len(self.values), 2))

            for value_index, value in enumerate(self.values):

                mean_enthalpy = value[0].value.to(default_unit).magnitude
                sem_enthalpy = value[0].uncertainty.to(default_unit).magnitude

                mean_free_energy = value[1].value.to(default_unit).magnitude
                sem_free_energy = value[1].uncertainty.to(default_unit).magnitude

                sampled_enthalpy = np.random.normal(mean_enthalpy, sem_enthalpy) * default_unit
                sampled_free_energy = np.random.normal(mean_free_energy, sem_free_energy) * default_unit

                cycle_values[value_index][0] = sampled_enthalpy.to(default_unit).magnitude
                cycle_values[value_index][1] = (-beta * sampled_free_energy).to(unit.dimensionless).magnitude

            #      Σ_{n} [ ΔH_{n} × exp(-βΔG°_{n}) ]
            # ΔH = ---------------------------------
            #            Σ_{n} exp(-βΔG°_{n})

            cycle_result[cycle_index] = np.sum(cycle_values[:, 0] * np.exp(cycle_values[:, 1])) \
                                        / np.sum(np.exp(cycle_values[:, 1]))

        mean = np.mean(cycle_result) * default_unit
        sem = np.std(cycle_result) * default_unit

        confidence_intervals = np.empty(2)
        sorted_statistics = np.sort(cycle_result)
        confidence_intervals[0] = sorted_statistics[int(0.025 * self.bootstrap_cycles)]
        confidence_intervals[1] = sorted_statistics[int(0.975 * self.bootstrap_cycles)]

        return mean, sem, confidence_intervals * default_unit
