"""
A collection of protocols for running analysing the results of molecular simulations.
"""

import logging
from os import path

import numpy as np

from propertyestimator import unit
from propertyestimator.utils import statistics, timeseries
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, bootstrap
from propertyestimator.workflow.decorators import protocol_input, protocol_output, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.thermodynamics import ThermodynamicState


from propertyestimator.protocols.miscellaneous import AddValues

@register_calculation_protocol()
class AddBindingFreeEnergies(AddValues):
    """A protocol to add together a list of binding free energies.

    Notes
    -----
    The `values` input must either be a list of unit.Quantity, a ProtocolPath to a list
    of unit.Quantity, or a list of ProtocolPath which each point to a unit.Quantity.
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

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._values = None
        self._thermodynamic_state = None
        self._result = None

    def execute(self, directory, available_resources):

        results_dictionary = self.bootstrap()
        self._result = results_dictionary["mean"]

        return self._get_output_dictionary()

    def bootstrap(self, cycles=1000, with_replacement=True):
        R = (1 * unit.molar_gas_constant).to(unit.kilocalorie / unit.mole / unit.kelvin)
        T = self.thermodynamic_state.temperature
        beta = 1.0 / (R * T)

        cycle_result = np.empty(cycles)
        for cycle_index, cycle in enumerate(range(cycles)):
            cycle_values = np.empty(len(self.values))
            for value_index, value in enumerate(self.values):

                magnitude, sem, units = _workaround_pint(value)
                cycle_values[value_index] = np.random.normal(magnitude, sem)

            cycle_values *= units
            magnitude, sem, units = _workaround_pint(-R * T * np.log(np.sum(np.exp(-beta * cycle_values))))
            cycle_result[cycle_index] = magnitude

        cycle_result *= units
        mean = np.mean(cycle_result)
        sem = np.std(cycle_result)

        ci = np.empty((2))
        sorted_statistics = np.sort(cycle_result)
        ci[0] = sorted_statistics[int(0.025 * cycles)]
        ci[1] = sorted_statistics[int(0.985 * cycles)]

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


def _workaround_pint(quantity):
    # Work around https://github.com/hgrecco/pint/issues/484

    from pint.measurement import _Measurement
    magnitude = quantity.value.magnitude
    if isinstance(quantity, _Measurement):
        uncertainty = quantity.error.magnitude
    else:
        uncertainty = None
    units = quantity.units

    return magnitude, uncertainty, units
