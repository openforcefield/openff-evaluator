"""
A collection of protocols for calculating the gradients of observables
with respect to force field parameters.
"""
import abc
import typing

import pint

from propertyestimator.attributes import UNDEFINED
from propertyestimator.forcefield import ParameterGradient, ParameterGradientKey
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import Protocol


@workflow_protocol()
class BaseGradientPotentials(Protocol, abc.ABC):
    """A base class for protocols which will evaluate the reduced potentials of a
    series of configurations using a set of force field parameters which have been
    slightly increased and slightly decreased. These are mainly useful when
    estimating gradients with respect to force field parameters using the central
    difference method.
    """

    force_field_path = InputAttribute(
        docstring="The path to the force field which contains the parameters to "
        "differentiate the observable with respect to. When reweighting "
        "observables, this should be the `target` force field.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    statistics_path = InputAttribute(
        docstring="The path to a statistics array containing potentials "
        "evaluated at each frame of the trajectory using the input "
        "`force_field_path` and at the input `thermodynamic_state`.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic state to estimate the gradients at. When "
        "reweighting observables, this should be the `target` state.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    substance = InputAttribute(
        docstring="The substance which describes the composition of the system.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    coordinate_file_path = InputAttribute(
        docstring="A path to a PDB coordinate file which describes the topology of "
        "the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    trajectory_file_path = InputAttribute(
        docstring="A path to the trajectory of configurations",
        type_hint=str,
        default_value=UNDEFINED,
    )

    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled when "
        "re-evaluating the reduced potentials.",
        type_hint=bool,
        default_value=True,
    )

    parameter_key = InputAttribute(
        docstring="The key of the parameter to differentiate with respect to.",
        type_hint=ParameterGradientKey,
        default_value=UNDEFINED,
    )
    perturbation_scale = InputAttribute(
        docstring="The amount to perturb the parameter by, such that "
        "p_new = p_old * (1 +/- `perturbation_scale`)",
        type_hint=float,
        default_value=1.0e-4,
    )

    use_subset_of_force_field = InputAttribute(
        docstring="If true, the reduced potentials will be estimated using "
        "a system which only contains the parameters of interest, e.g. if the "
        "gradient of interest is with respect to the VdW epsilon parameter, then "
        "all valence / electrostatic terms will be ignored.",
        type_hint=bool,
        default_value=True,
    )

    effective_sample_indices = InputAttribute(
        docstring="This a placeholder input which is not currently implemented.",
        type_hint=list,
        default_value=UNDEFINED,
        optional=True,
    )

    reverse_potentials_path = OutputAttribute(
        docstring="A file path to the energies evaluated using the parameters"
        "perturbed in the reverse direction.",
        type_hint=str,
    )
    forward_potentials_path = OutputAttribute(
        docstring="A file path to the energies evaluated using the parameters"
        "perturbed in the forward direction.",
        type_hint=str,
    )
    reverse_parameter_value = OutputAttribute(
        docstring="The value of the parameter perturbed in the reverse direction.",
        type_hint=pint.Quantity,
    )
    forward_parameter_value = OutputAttribute(
        docstring="The value of the parameter perturbed in the forward direction.",
        type_hint=pint.Quantity,
    )


@workflow_protocol()
class CentralDifferenceGradient(Protocol):
    """A protocol which employs the central diference method
    to estimate the gradient of an observable A, such that

    grad = (A(x-h) - A(x+h)) / (2h)

    Notes
    -----
    The `values` input must either be a list of pint.Quantity, a ProtocolPath to a list
    of pint.Quantity, or a list of ProtocolPath which each point to a pint.Quantity.
    """

    parameter_key = InputAttribute(
        docstring="The key of the parameter to differentiate with respect to.",
        type_hint=ParameterGradientKey,
        default_value=UNDEFINED,
    )

    reverse_observable_value = InputAttribute(
        docstring="The value of the observable evaluated using the parameters"
        "perturbed in the reverse direction.",
        type_hint=typing.Union[pint.Quantity, pint.Measurement],
        default_value=UNDEFINED,
    )
    forward_observable_value = InputAttribute(
        docstring="The value of the observable evaluated using the parameters"
        "perturbed in the forward direction.",
        type_hint=typing.Union[pint.Quantity, pint.Measurement],
        default_value=UNDEFINED,
    )

    reverse_parameter_value = InputAttribute(
        docstring="The value of the parameter perturbed in the reverse direction.",
        type_hint=pint.Quantity,
        default_value=UNDEFINED,
    )
    forward_parameter_value = InputAttribute(
        docstring="The value of the parameter perturbed in the forward direction.",
        type_hint=pint.Quantity,
        default_value=UNDEFINED,
    )

    gradient = OutputAttribute(
        docstring="The estimated gradient", type_hint=ParameterGradient
    )

    def _execute(self, directory, available_resources):

        if self.forward_parameter_value < self.reverse_parameter_value:

            raise ValueError(
                f"The forward parameter value ({self.forward_parameter_value}) must "
                f"be larger than the reverse value ({self.reverse_parameter_value})."
            )

        reverse_value = self.reverse_observable_value
        forward_value = self.forward_observable_value

        if isinstance(reverse_value, pint.Measurement):
            reverse_value = reverse_value.value

        if isinstance(forward_value, pint.Measurement):
            forward_value = forward_value.value

        gradient = (forward_value - reverse_value) / (
            self.forward_parameter_value - self.reverse_parameter_value
        )

        self.gradient = ParameterGradient(self.parameter_key, gradient)
