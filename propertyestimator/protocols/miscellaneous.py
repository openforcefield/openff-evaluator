"""
A collection of miscellaneous protocols, mostly aimed at performing simple
math operations.
"""
import typing

import numpy as np
from propertyestimator import unit
from propertyestimator.properties import ParameterGradient
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
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

    @protocol_output(typing.Union[int, float, EstimatedQuantity, unit.Quantity, ParameterGradient])
    def result(self):
        """The sum of the values."""
        pass

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

    @protocol_input(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def value_a(self):
        """`value_a` in the formula `result = value_b - value_a`"""
        pass

    @protocol_input(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def value_b(self):
        """`value_b` in the formula  `result = value_b - value_a`"""
        pass

    @protocol_output(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
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
class MultiplyValue(BaseProtocol):
    """A protocol which multiplies a value by a specified scalar
    """

    @protocol_input(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def value(self):
        """The value to multiply."""
        pass

    @protocol_input(typing.Union[int, float, unit.Quantity])
    def multiplier(self):
        """The scalar to multiply by."""
        pass

    @protocol_output(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def result(self):
        """The result of the multiplication."""
        pass

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

    @protocol_input(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def value(self):
        """The value to divide."""
        pass

    @protocol_input(typing.Union[int, float, unit.Quantity])
    def divisor(self):
        """The scalar to divide by."""
        pass

    @protocol_output(typing.Union[int, float, unit.Quantity, EstimatedQuantity, ParameterGradient])
    def result(self):
        """The result of the division."""
        pass

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
    @protocol_input(Substance)
    def component(self, value):
        """The component (e.g water) to which this value belongs."""
        pass

    @protocol_input(Substance)
    def full_substance(self, value):
        """The full substance of which the component of interest is a part."""
        pass

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
    @protocol_input(typing.Union[float, int, EstimatedQuantity, unit.Quantity, ParameterGradient])
    def value(self):
        """The value to be weighted."""
        pass

    @protocol_output(typing.Union[float, int, EstimatedQuantity, unit.Quantity, ParameterGradient])
    def weighted_value(self, value):
        """The value weighted by the `component`s mole fraction as determined from
        the `full_substance`."""
        pass

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

    @protocol_input(Substance)
    def input_substance(self):
        """The substance to filter."""
        pass

    @protocol_input(list)
    def component_roles(self):
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
        self._component_roles = None

        self._expected_components = -1

        self._filtered_substance = None

    def execute(self, directory, available_resources):

        filtered_components = []
        total_mole_fraction = 0.0

        for component in self._input_substance.components:

            if component.role not in self._component_roles:
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


@register_calculation_protocol()
class AddBindingFreeEnergies(AddValues):
    """A protocol to add together a list of binding free energies.

    Notes
    -----
    The `values` input must either be a list of EstimatedQuantity, a ProtocolPath to a list
    of EstimatedQuantity, or a list of ProtocolPath which each point to a EstimatedQuantity.
    """

    @protocol_input(list)
    def values(self):
        """The values to add together."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state at which the free energies were measured."""
        pass

    @protocol_input(int)
    def bootstrap_cycles(self):
        """The number of bootstrap cycles to perform when estimating
        the uncertainty in the combined free energies."""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

    @protocol_output(unit.Quantity)
    def confidence_intervals(self):
        """The confidence intervals on the summed free energy."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddBindingFreeEnergies object."""
        super().__init__(protocol_id)

        self._values = None
        self._thermodynamic_state = None
        self._bootstrap_cycles = 1000

        self._result = None
        self._confidence_intervals = None

    def execute(self, directory, available_resources):

        mean, uncertainty, confidence_intervals = self.bootstrap()

        self._result = EstimatedQuantity(mean,
                                         uncertainty,
                                         self._id)

        self._confidence_intervals = confidence_intervals

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

            cycle_values = np.empty(len(self._values))

            for value_index, value in enumerate(self._values):

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
class AddBindingEnthalpies(AddValues):
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

    @protocol_input(list)
    def enthalpy_free_energy_tuple(self):
        """The enthalpies to add together, passed as a tuple with their respective binding free energies."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state at which the free energies were measured."""
        pass

    @protocol_input(int)
    def bootstrap_cycles(self):
        """The number of bootstrap cycles to perform when estimating
        the uncertainty in the combined free energies."""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

    @protocol_output(unit.Quantity)
    def confidence_intervals(self):
        """The confidence intervals on the summed enthalpy."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddBindingEnthalpies object."""
        super().__init__(protocol_id)

        self._values = None
        self._thermodynamic_state = None
        self._bootstrap_cycles = 1000

        self._result = None
        self._confidence_intervals = None

        self._enthalpy_free_energy_tuple = None

    def execute(self, directory, available_resources):

        mean, uncertainty, confidence_intervals = self.bootstrap()

        self._result = EstimatedQuantity(mean,
                                         uncertainty,
                                         self._id)

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

        cycle_result = np.empty(self._bootstrap_cycles)

        for cycle_index, cycle in enumerate(range(self._bootstrap_cycles)):

            cycle_values = np.empty((len(self._values), 2))

            for value_index, value in enumerate(self._values):

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
        confidence_intervals[0] = sorted_statistics[int(0.025 * self._bootstrap_cycles)]
        confidence_intervals[1] = sorted_statistics[int(0.975 * self._bootstrap_cycles)]

        return mean, sem, confidence_intervals * default_unit