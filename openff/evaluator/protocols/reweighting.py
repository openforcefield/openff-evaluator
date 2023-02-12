"""
A collection of protocols for reweighting cached simulation data.
"""
import abc
import copy
import functools
import typing
from os import path

import numpy as np
import pymbar
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import ParameterGradient
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.analysis import compute_dielectric_constant
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.observables import (
    Observable,
    ObservableArray,
    ObservableFrame,
    bootstrap,
)
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


@workflow_protocol()
class ConcatenateTrajectories(Protocol):
    """A protocol which concatenates multiple trajectories into
    a single one.
    """

    input_coordinate_paths = InputAttribute(
        docstring="A list of paths to the starting PDB coordinates for each of the "
        "trajectories.",
        type_hint=list,
        default_value=UNDEFINED,
    )
    input_trajectory_paths = InputAttribute(
        docstring="A list of paths to the trajectories to concatenate.",
        type_hint=list,
        default_value=UNDEFINED,
    )

    output_coordinate_path = OutputAttribute(
        docstring="The path the PDB coordinate file which contains the topology "
        "of the concatenated trajectory.",
        type_hint=str,
    )

    output_trajectory_path = OutputAttribute(
        docstring="The path to the concatenated trajectory.", type_hint=str
    )

    def _execute(self, directory, available_resources):
        import mdtraj

        if len(self.input_coordinate_paths) != len(self.input_trajectory_paths):
            raise ValueError(
                "There should be the same number of coordinate and trajectory paths."
            )

        if len(self.input_trajectory_paths) == 0:
            raise ValueError("No trajectories were given to concatenate.")

        trajectories = []

        output_coordinate_path = None

        for coordinate_path, trajectory_path in zip(
            self.input_coordinate_paths, self.input_trajectory_paths
        ):
            output_coordinate_path = output_coordinate_path or coordinate_path
            trajectories.append(mdtraj.load_dcd(trajectory_path, coordinate_path))

        self.output_coordinate_path = output_coordinate_path
        output_trajectory = (
            trajectories[0]
            if len(trajectories) == 1
            else mdtraj.join(trajectories, False, False)
        )

        self.output_trajectory_path = path.join(directory, "output_trajectory.dcd")
        output_trajectory.save_dcd(self.output_trajectory_path)


@workflow_protocol()
class ConcatenateObservables(Protocol):
    """A protocol which concatenates multiple ``ObservableFrame`` objects into
    a single ``ObservableFrame`` object.
    """

    input_observables = InputAttribute(
        docstring="A list of observable arrays to concatenate.",
        type_hint=list,
        default_value=UNDEFINED,
    )
    output_observables = OutputAttribute(
        docstring="The concatenated observable array.",
        type_hint=typing.Union[ObservableArray, ObservableFrame],
    )

    def _execute(self, directory, available_resources):
        if len(self.input_observables) == 0:
            raise ValueError("No arrays were given to concatenate.")

        if not all(
            isinstance(observables, type(self.input_observables[0]))
            for observables in self.input_observables
        ):
            raise ValueError("The observables to concatenate must be the same type.")

        object_type = type(self.input_observables[0])

        if len(self.input_observables) > 1:
            self.output_observables = object_type.join(*self.input_observables)
        else:
            self.output_observables = copy.deepcopy(self.input_observables[0])


class BaseEvaluateEnergies(Protocol, abc.ABC):
    """A base class for protocols which will re-evaluate the energy of a series
    of configurations for a given set of force field parameters.
    """

    thermodynamic_state = InputAttribute(
        docstring="The state to calculate the reduced potentials at.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    parameterized_system = InputAttribute(
        docstring="The parameterized system object which encodes the systems potential "
        "energy function.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )
    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled.",
        type_hint=bool,
        default_value=True,
    )

    trajectory_file_path = InputAttribute(
        docstring="The path to the trajectory file which contains the "
        "configurations to calculate the energies of.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    gradient_parameters = InputAttribute(
        docstring="An optional list of parameters to differentiate the evaluated "
        "energies with respect to.",
        type_hint=list,
        default_value=lambda: list(),
    )

    output_observables = OutputAttribute(
        docstring="An observable array which stores the reduced potentials potential "
        "energies evaluated at the specified state and using the specified system "
        "object for each configuration in the trajectory.",
        type_hint=ObservableFrame,
    )


class BaseMBARProtocol(Protocol, abc.ABC):
    """Re-weights a set of observables using MBAR to calculate
    the average value of the observables at a different state
    than they were originally measured.
    """

    reference_reduced_potentials: typing.List[ObservableArray] = InputAttribute(
        docstring="The reduced potentials of each configuration evaluated at each of "
        "the reference states.",
        type_hint=list,
        default_value=UNDEFINED,
    )
    target_reduced_potentials = InputAttribute(
        docstring="The reduced potentials of each configuration evaluated at the "
        "target state.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
    )

    frame_counts = InputAttribute(
        docstring="The number of configurations per reference state. The sum of these"
        "should equal the length of the ``reference_reduced_potentials`` and "
        "``target_reduced_potentials`` input arrays as well any input observable "
        "arrays.",
        type_hint=list,
        default_value=UNDEFINED,
    )

    bootstrap_uncertainties = InputAttribute(
        docstring="If true, bootstrapping will be used to estimated the total "
        "uncertainty in the reweighted value.",
        type_hint=bool,
        default_value=False,
    )
    bootstrap_iterations = InputAttribute(
        docstring="The number of bootstrap iterations to perform if bootstraped "
        "uncertainties have been requested",
        type_hint=int,
        default_value=250,
    )

    required_effective_samples = InputAttribute(
        docstring="The minimum number of effective samples required to be able to "
        "reweight the observable. If the effective samples is less than this minimum "
        "an exception will be raised.",
        type_hint=int,
        default_value=50,
    )

    value = OutputAttribute(
        docstring="The re-weighted average value of the observable at the target "
        "state.",
        type_hint=Observable,
    )
    effective_samples = OutputAttribute(
        docstring="The number of effective samples which were re-weighted.",
        type_hint=float,
    )

    @abc.abstractmethod
    def _observables(self) -> typing.Dict[str, ObservableArray]:
        """The observables which will be re-weighted to yield the final average
        observable of interest.
        """
        raise NotImplementedError()

    @staticmethod
    def _compute_weights(
        mbar: pymbar.MBAR, target_reduced_potentials: ObservableArray
    ) -> ObservableArray:
        """Return the values that each sample at the target state should be weighted
        by.

        Parameters
        ----------
        mbar
            A pre-computed MBAR object encoded information from the reference states.
        target_reduced_potentials
            The reduced potentials at the target state.

        Returns
        -------
            The values to weight each sample by.
        """
        from scipy.special import logsumexp

        u_kn = target_reduced_potentials.value.to(unit.dimensionless).magnitude.T

        log_denominator_n = logsumexp(mbar.f_k - mbar.u_kn.T, b=mbar.N_k, axis=1)

        f_hat = -logsumexp(-u_kn - log_denominator_n, axis=1)

        # Calculate the weights
        weights = np.exp(f_hat - u_kn - log_denominator_n) * unit.dimensionless

        # Compute the gradients of the weights.
        weight_gradients = []

        for gradient in target_reduced_potentials.gradients:
            gradient_value = gradient.value.magnitude.flatten()

            # Compute the numerator of the gradient. We need to specifically ask for the
            # sign of the exp sum as the numerator may be negative.
            d_f_hat_numerator, d_f_hat_numerator_sign = logsumexp(
                -u_kn - log_denominator_n, b=gradient_value, axis=1, return_sign=True
            )
            d_f_hat_d_theta = d_f_hat_numerator_sign * np.exp(d_f_hat_numerator + f_hat)

            d_weights_d_theta = (
                (d_f_hat_d_theta - gradient_value) * weights * gradient.value.units
            )

            weight_gradients.append(
                ParameterGradient(key=gradient.key, value=d_weights_d_theta.T)
            )

        return ObservableArray(value=weights.T, gradients=weight_gradients)

    def _compute_effective_samples(
        self, reference_reduced_potentials: ObservableArray
    ) -> float:
        """Compute the effective number of samples which contribute to the final
        re-weighted estimate.

        Parameters
        ----------
        reference_reduced_potentials
            An 2D array containing the reduced potentials of each configuration
            evaluated at each reference state.

        Returns
        -------
            The effective number of samples.
        """

        # Construct an MBAR object so that the number of effective samples can
        # be computed.
        mbar = pymbar.MBAR(
            reference_reduced_potentials.value.to(unit.dimensionless).magnitude.T,
            self.frame_counts,
            verbose=False,
            relative_tolerance=1e-12,
        )

        weights = (
            self._compute_weights(mbar, self.target_reduced_potentials)
            .value.to(unit.dimensionless)
            .magnitude
        )

        effective_samples = 1.0 / np.sum(weights**2)
        return float(effective_samples)

    def _execute(self, directory, available_resources):
        # Retrieve the observables to reweight.
        observables = self._observables()

        if len(observables) == 0:
            raise ValueError("There were no observables to reweight.")

        if len(self.frame_counts) != len(self.reference_reduced_potentials):
            raise ValueError("A frame count must be provided for each reference state.")

        expected_frames = sum(self.frame_counts)

        if any(
            len(input_array) != expected_frames
            for input_array in [
                self.target_reduced_potentials,
                *self.reference_reduced_potentials,
                *observables.values(),
            ]
        ):
            raise ValueError(
                f"The length of the input arrays do not match the expected length "
                f"specified by the frame counts ({expected_frames})."
            )

        # Concatenate the reduced reference potentials into a single array.
        # We ignore the gradients of the reference state potential as these
        # should be all zero.
        reference_reduced_potentials = ObservableArray(
            value=np.hstack(
                [
                    reduced_potentials.value
                    for reduced_potentials in self.reference_reduced_potentials
                ]
            )
        )

        # Ensure that there is enough effective samples to re-weight.
        self.effective_samples = self._compute_effective_samples(
            reference_reduced_potentials
        )

        if self.effective_samples < self.required_effective_samples:
            raise ValueError(
                f"There was not enough effective samples to reweight - "
                f"{self.effective_samples} < {self.required_effective_samples}"
            )

        if self.bootstrap_uncertainties:
            self.value = bootstrap(
                self._bootstrap_function,
                self.bootstrap_iterations,
                1.0,
                self.frame_counts,
                reference_reduced_potentials=reference_reduced_potentials,
                target_reduced_potentials=self.target_reduced_potentials,
                **observables,
            )

        else:
            self.value = self._bootstrap_function(
                reference_reduced_potentials=reference_reduced_potentials,
                target_reduced_potentials=self.target_reduced_potentials,
                **observables,
            )

    def _bootstrap_function(self, **observables: ObservableArray) -> Observable:
        """Re-weights a set of reference observables to the target state.

        Parameters
        -------
        observables
            The observables to reweight, in addition to the reference and target
            reduced potentials.
        """

        reference_reduced_potentials = observables.pop("reference_reduced_potentials")
        target_reduced_potentials = observables.pop("target_reduced_potentials")

        # Construct the mbar object using the specified reference reduced potentials.
        # These may be the input values or values which have been sampled during
        # bootstrapping, hence why it is not precomputed once.
        mbar = pymbar.MBAR(
            reference_reduced_potentials.value.to(unit.dimensionless).magnitude.T,
            self.frame_counts,
            verbose=False,
            relative_tolerance=1e-12,
        )

        # Compute the MBAR weights.
        weights = self._compute_weights(mbar, target_reduced_potentials)

        return self._reweight_observables(
            weights, mbar, target_reduced_potentials, **observables
        )

    def _reweight_observables(
        self,
        weights: ObservableArray,
        mbar: pymbar.MBAR,
        target_reduced_potentials: ObservableArray,
        **observables: ObservableArray,
    ) -> typing.Union[ObservableArray, Observable]:
        """A function which computes the average value of an observable using
        weights computed from MBAR and from a set of component observables.

        Parameters
        ----------
        weights
            The MBAR weights
        observables
            The component observables which may be combined to yield the final
            average observable of interest.
        mbar
            A pre-computed MBAR object encoded information from the reference states.
            This will be used to compute the std error when not bootstrapping.
        target_reduced_potentials
            The reduced potentials at the target state. This will be used to compute
            the std error when not bootstrapping.

        Returns
        -------
            The re-weighted average observable.
        """

        observable = observables.pop("observable")
        assert len(observables) == 0

        return_type = ObservableArray if observable.value.shape[1] > 1 else Observable

        weighted_observable = weights * observable

        average_value = weighted_observable.value.sum(axis=0)
        average_gradients = [
            ParameterGradient(key=gradient.key, value=gradient.value.sum(axis=0))
            for gradient in weighted_observable.gradients
        ]

        if return_type == Observable:
            average_value = average_value.item()
            average_gradients = [
                ParameterGradient(key=gradient.key, value=gradient.value.item())
                for gradient in average_gradients
            ]

        else:
            average_value = average_value.reshape(1, -1)
            average_gradients = [
                ParameterGradient(key=gradient.key, value=gradient.value.reshape(1, -1))
                for gradient in average_gradients
            ]

        if self.bootstrap_uncertainties is False:
            # Unfortunately we need to re-compute the average observable for now
            # as pymbar does not expose an easier way to compute the average
            # uncertainty.
            observable_dimensions = observable.value.shape[1]
            assert observable_dimensions == 1

            from openff.evaluator.utils.pymbar import compute_expectations

            results = getattr(mbar, compute_expectations)(
                observable.value.T.magnitude,
                target_reduced_potentials.value.T.magnitude,
                state_dependent=True,
            )

            uncertainty = results[1][-1] * observable.value.units
            average_value = average_value.plus_minus(uncertainty)

        return return_type(value=average_value, gradients=average_gradients)


@workflow_protocol()
class ReweightObservable(BaseMBARProtocol):
    """Reweight an array of observables to a new state using MBAR."""

    observable = InputAttribute(
        docstring="The observables to reweight. The array should contain the values of "
        "the observable evaluated for of each configuration at the target state.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
    )

    def _observables(self) -> typing.Dict[str, ObservableArray]:
        return {"observable": self.observable}


@workflow_protocol()
class ReweightDielectricConstant(BaseMBARProtocol):
    """Computes the avergage value of the dielectric constant be re-weighting a set
    a set of dipole moments and volumes using MBAR.
    """

    dipole_moments = InputAttribute(
        docstring="The dipole moments evaluated at reference state's configurations"
        "using the force field of the target state.",
        type_hint=typing.Union[ObservableArray, list],
        default_value=UNDEFINED,
    )
    volumes = InputAttribute(
        docstring="The dipole moments evaluated at reference state's configurations"
        "using the force field of the target state.",
        type_hint=typing.Union[ObservableArray, list],
        default_value=UNDEFINED,
    )

    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic state to re-weight to.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    def __init__(self, protocol_id):
        super().__init__(protocol_id)
        self.bootstrap_uncertainties = True

    def _observables(self) -> typing.Dict[str, ObservableArray]:
        return {"volumes": self.volumes, "dipole_moments": self.dipole_moments}

    def _reweight_observables(
        self,
        weights: ObservableArray,
        mbar: pymbar.MBAR,
        target_reduced_potentials: ObservableArray,
        **observables: ObservableArray,
    ) -> Observable:
        volumes = observables.pop("volumes")
        dipole_moments = observables.pop("dipole_moments")

        dielectric_constant = compute_dielectric_constant(
            dipole_moments,
            volumes,
            self.thermodynamic_state.temperature,
            functools.partial(
                super(ReweightDielectricConstant, self)._reweight_observables,
                weights=weights,
                mbar=mbar,
                target_reduced_potentials=target_reduced_potentials,
            ),
        )

        return dielectric_constant

    def _execute(self, directory, available_resources):
        if not self.bootstrap_uncertainties:
            raise ValueError(
                "The uncertainty in the average dielectric constant should only be "
                "computed using bootstrapping."
            )

        super(ReweightDielectricConstant, self)._execute(directory, available_resources)
