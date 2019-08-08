"""
A collection of protocols for running analysing the results of molecular simulations.
"""

import numpy as np

from propertyestimator import unit
from propertyestimator.protocols.miscellaneous import AddValues
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol


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

        self._result = None
        self._confidence_intervals = None

    def execute(self, directory, available_resources):

        results_dictionary = self.bootstrap()

        self._result = EstimatedQuantity(results_dictionary['mean'],
                                         results_dictionary['sem'],
                                         self._id)

        self._confidence_intervals = results_dictionary['ci']

        return self._get_output_dictionary()

    def bootstrap(self, cycles=1000, with_replacement=True):

        default_unit = unit.kilocalorie / unit.mole

        boltzmann_factor = self.thermodynamic_state.temperature * unit.molar_gas_constant
        boltzmann_factor.ito(default_unit)

        beta = 1.0 / boltzmann_factor

        cycle_result = np.empty(cycles)

        for cycle_index, cycle in enumerate(range(cycles)):

            cycle_values = np.empty(len(self._values))

            for value_index, value in enumerate(self._values):

                mean = value.value.to(default_unit).magnitude
                sem = value.uncertainty.to(default_unit).magnitude

                sampled_value = np.random.normal(mean, sem) * default_unit
                cycle_values[value_index] = (-beta * sampled_value).to(unit.dimensionless).magnitude

            cycle_result[cycle_index] = np.log(np.sum(np.exp(cycle_values)))

        mean = np.mean(-boltzmann_factor * cycle_result)
        sem = np.std(-boltzmann_factor * cycle_result)

        ci = np.empty((2))
        sorted_statistics = np.sort(cycle_result)
        ci[0] = sorted_statistics[int(0.025 * cycles)]
        ci[1] = sorted_statistics[int(0.985 * cycles)]

        ci = -boltzmann_factor * ci

        results = {"mean": mean,
                   "sem": sem,
                   "ci": ci}
        # Simon: say I also want to store the confidence intervals here, how would I do it?

        return results


@register_calculation_protocol()
class AddBindingEnthalpies(AddValues):

    @protocol_input(list)
    def values(self):
        """The values to add together."""
        pass

    @protocol_input(list)
    def values(self):
        """The values to add together."""
        pass
