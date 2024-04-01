"""
A collection of protocols for calculating the gradients of observables with respect to
force field parameters.
"""

import abc
from typing import Union

import numpy
from openff.units import unit
from openff.units.openmm import from_openmm

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import (
    ForceFieldSource,
    ParameterGradient,
    SmirnoffForceFieldSource,
)
from openff.evaluator.utils.observables import Observable, ObservableArray
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


@workflow_protocol()
class ZeroGradients(Protocol, abc.ABC):
    """Zeros the gradients of an observable with respect to a specified set of force
    field parameters.
    """

    input_observables = InputAttribute(
        docstring="The observable to set the gradients of.",
        type_hint=Union[Observable, ObservableArray],
        default_value=UNDEFINED,
    )

    force_field_path = InputAttribute(
        docstring="The path to the force field which contains the parameters to "
        "differentiate the observable with respect to. This is many used to get the "
        "correct units for the parameters.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    gradient_parameters = InputAttribute(
        docstring="The parameters to zero the gradient with respect to.",
        type_hint=list,
        default_value=lambda: list(),
    )

    output_observables = OutputAttribute(
        docstring="The observable with zeroed gradients.",
        type_hint=Union[Observable, ObservableArray],
    )

    def _execute(self, directory, available_resources):
        from openmm import unit as openmm_unit

        force_field_source = ForceFieldSource.from_json(self.force_field_path)

        if not isinstance(force_field_source, SmirnoffForceFieldSource):
            raise ValueError("Only SMIRNOFF force fields are supported.")

        force_field = force_field_source.to_force_field()

        def _get_parameter_unit(gradient_key):
            parameter = force_field.get_parameter_handler(gradient_key.tag)

            if gradient_key.smirks is not None:
                parameter = parameter.parameters[gradient_key.smirks]

            value = getattr(parameter, gradient_key.attribute)

            if isinstance(value, openmm_unit.Quantity):
                return from_openmm(value).units

            return unit.dimensionless

        parameter_units = {
            gradient_key: _get_parameter_unit(gradient_key)
            for gradient_key in self.gradient_parameters
        }

        self.input_observables.clear_gradients()

        if isinstance(self.input_observables, Observable):
            self.output_observables = Observable(
                value=self.input_observables.value,
                gradients=[
                    ParameterGradient(
                        key=gradient_key,
                        value=(
                            0.0
                            * self.input_observables.value.units
                            / parameter_units[gradient_key]
                        ),
                    )
                    for gradient_key in self.gradient_parameters
                ],
            )

        elif isinstance(self.input_observables, ObservableArray):
            self.output_observables = ObservableArray(
                value=self.input_observables.value,
                gradients=[
                    ParameterGradient(
                        key=gradient_key,
                        value=(
                            numpy.zeros(self.input_observables.value.shape)
                            * self.input_observables.value.units
                            / parameter_units[gradient_key]
                        ),
                    )
                    for gradient_key in self.gradient_parameters
                ],
            )

        else:
            raise NotImplementedError()
