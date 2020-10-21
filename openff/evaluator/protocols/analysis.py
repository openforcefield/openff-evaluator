"""
A collection of protocols for running analysing the results of molecular simulations.
"""
import abc
import itertools
import typing
from os import path

import numpy as np
import pint

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import ParameterGradient, SmirnoffForceFieldSource
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import timeseries
from openff.evaluator.utils.observables import (
    Observable,
    ObservableArray,
    ObservableFrame,
    bootstrap,
)
from openff.evaluator.utils.openmm import openmm_quantity_to_pint, system_subset
from openff.evaluator.utils.timeseries import (
    TimeSeriesStatistics,
    analyze_time_series,
    get_uncorrelated_indices,
)
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)

if typing.TYPE_CHECKING:
    from simtk import openmm


E0 = 8.854187817e-12 * unit.farad / unit.meter  # Taken from QCElemental


def compute_dielectric_constant(
    dipole_moments: ObservableArray,
    volumes: ObservableArray,
    temperature: pint.Quantity,
    average_function,
) -> Observable:
    """A function to compute the average dielectric constant from an array of
    dipole moments and an array of volumes, whereby the average values of the
    observables are computed using a custom function.

    Parameters
    ----------
    dipole_moments
        The dipole moments array.
    volumes
        The volume array.
    temperature
        The temperature at which the dipole_moments and volumes were sampled.
    average_function
        The function to use when evaluating the average of an observable.

    Returns
    -------
        The average value of the dielectric constant.
    """

    dipole_moments_sqr = dipole_moments * dipole_moments
    dipole_moments_sqr = ObservableArray(
        value=dipole_moments_sqr.value.sum(axis=1),
        gradients=[
            ParameterGradient(gradient.key, gradient.value.sum(axis=1))
            for gradient in dipole_moments_sqr.gradients
        ],
    )

    avg_sqr_dipole_moments = average_function(observable=dipole_moments_sqr)
    avg_sqr_dipole_moments = ObservableArray(
        avg_sqr_dipole_moments.value, avg_sqr_dipole_moments.gradients
    )

    avg_dipole_moment = average_function(observable=dipole_moments)

    avg_dipole_moment_sqr = avg_dipole_moment * avg_dipole_moment
    avg_dipole_moment_sqr = ObservableArray(
        value=avg_dipole_moment_sqr.value.sum(axis=1),
        gradients=[
            ParameterGradient(gradient.key, gradient.value.sum(axis=1))
            for gradient in avg_dipole_moment_sqr.gradients
        ],
    )

    avg_volume = average_function(observable=volumes)
    avg_volume = ObservableArray(avg_volume.value, avg_volume.gradients)

    dipole_variance = avg_sqr_dipole_moments - avg_dipole_moment_sqr

    prefactor = 1.0 / (3.0 * E0 * unit.boltzmann_constant * temperature)

    dielectric_constant = 1.0 * unit.dimensionless + prefactor * (
        dipole_variance / avg_volume
    )

    return Observable(
        value=dielectric_constant.value.item().to(unit.dimensionless),
        gradients=[
            ParameterGradient(
                gradient.key,
                gradient.value.item().to(
                    unit.dimensionless
                    * gradient.value.units
                    / dielectric_constant.value.units
                ),
            )
            for gradient in dielectric_constant.gradients
        ],
    )


class BaseAverageObservable(Protocol, abc.ABC):
    """An abstract base class for protocols which will calculate the
    average value of an observable and its uncertainty via bootstrapping.
    """

    bootstrap_iterations = InputAttribute(
        docstring="The number of bootstrap iterations to perform.",
        type_hint=int,
        default_value=250,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
    )
    bootstrap_sample_size = InputAttribute(
        docstring="The relative sample size to use for bootstrapping.",
        type_hint=float,
        default_value=1.0,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
    )

    thermodynamic_state = InputAttribute(
        docstring="The state at which the observables were computed. This is required "
        "to compute ensemble averages of the gradients of the observable with respect "
        "to force field parameters.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
        optional=True,
    )
    potential_energies = InputAttribute(
        docstring="The potential energies which were evaluated at the same "
        "configurations and using the same force field parameters as the observable to "
        "average. This is required to compute ensemble averages of the gradients of "
        "the observable with respect to force field parameters.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
        optional=True,
    )

    value = OutputAttribute(
        docstring="The average value of the observable.", type_hint=Observable
    )

    time_series_statistics = OutputAttribute(
        docstring="Statistics about the observables from which the average was computed."
        "These include the statistical inefficiency and the index after which the "
        "observables have become stationary (i.e. equilibrated).",
        type_hint=TimeSeriesStatistics,
    )

    @abc.abstractmethod
    def _observables(self) -> typing.Dict[str, ObservableArray]:
        """A function which should return the observables to pass to the
        ``_bootstrap_function`` function."""
        raise NotImplementedError()

    def _bootstrap_function(self, **kwargs: ObservableArray) -> Observable:
        """The function to perform on the data set being sampled by
        bootstrapping.

        Parameters
        ----------
        observables
            The bootstrap sample values.

        Returns
        -------
            The result of evaluating the data.
        """

        # The simple base function only supports a single observable.
        assert len(kwargs) == 1

        # Compute the mean observable.
        sample_observable = next(iter(kwargs.values()))
        mean_observable = np.mean(sample_observable.value, axis=0)

        if sample_observable.value.shape[1] > 1:
            mean_observable = mean_observable.reshape(1, -1)
        else:
            mean_observable = mean_observable.item()

        # Retrieve the potential gradients for easy access
        potential_gradients = {
            gradient.key: gradient.value
            for gradient in (
                []
                if self.potential_energies == UNDEFINED
                else self.potential_energies.gradients
            )
        }
        observable_gradients = {
            gradient.key: gradient
            for gradient in (
                []
                if self.potential_energies == UNDEFINED
                else sample_observable.gradients
            )
        }

        # Compute the mean gradients.
        gradients = []

        for gradient_key in observable_gradients:

            gradient = observable_gradients[gradient_key]

            value = np.mean(gradient.value, axis=0) - self.thermodynamic_state.beta * (
                np.mean(
                    sample_observable.value * potential_gradients[gradient.key],
                    axis=0,
                )
                - (
                    np.mean(sample_observable.value, axis=0)
                    * np.mean(potential_gradients[gradient.key], axis=0)
                )
            )

            if sample_observable.value.shape[1] > 1:
                value = value.reshape(1, -1)
            else:
                value = value.item()

            gradients.append(ParameterGradient(key=gradient.key, value=value))

        return_type = (
            Observable if sample_observable.value.shape[1] == 1 else ObservableArray
        )
        return return_type(value=mean_observable, gradients=gradients)

    def _execute(self, directory, available_resources):

        # Retrieve the list of observables to compute the average using.
        observables = self._observables()

        if len(observables) == 0:
            raise ValueError("There are no observables to average.")

        expected_length = len(next(iter(observables.values())))

        if not any(len(observable) != expected_length for observable in observables):
            raise ValueError("The observables to average must have the same length.")

        if (
            self.potential_energies != UNDEFINED
            and self.thermodynamic_state == UNDEFINED
        ):
            raise ValueError(
                "The `thermodynamic_state` must be provided when the "
                "`potential_energies` input is specified (i.e. when gradients should "
                "be computed."
            )

        if self.potential_energies != UNDEFINED:

            potential_gradients = {
                gradient.key for gradient in self.potential_energies.gradients
            }

            if any(
                {gradient.key for gradient in observable.gradients}
                != potential_gradients
                for observable in observables.values()
            ):

                raise ValueError(
                    "The potential energies must have been differentiated with respect "
                    "to the same force field parameters as the observables of interest."
                )

        # Find the largest equilibration time and statistical inefficiency for each of
        # the observables.
        equilibration_index = -1
        statistical_inefficiency = 0.0

        for observable in observables.values():

            observable_statistics = analyze_time_series(observable.value.magnitude)

            equilibration_index = max(
                equilibration_index, observable_statistics.equilibration_index
            )
            statistical_inefficiency = max(
                statistical_inefficiency, observable_statistics.statistical_inefficiency
            )

        equilibration_index = int(equilibration_index)
        statistical_inefficiency = float(statistical_inefficiency)

        uncorrelated_indices = get_uncorrelated_indices(
            expected_length - equilibration_index, statistical_inefficiency
        )

        self.time_series_statistics = TimeSeriesStatistics(
            n_total_points=expected_length,
            n_uncorrelated_points=len(uncorrelated_indices),
            statistical_inefficiency=statistical_inefficiency,
            equilibration_index=equilibration_index,
        )

        # Decorrelate the observables.
        uncorrelated_observables = {
            key: observable.subset(
                [index + equilibration_index for index in uncorrelated_indices]
            )
            for key, observable in observables.items()
        }

        if self.potential_energies != UNDEFINED:

            self.potential_energies = self.potential_energies.subset(
                [index + equilibration_index for index in uncorrelated_indices]
            )

        self.value = bootstrap(
            self._bootstrap_function,
            self.bootstrap_iterations,
            self.bootstrap_sample_size,
            **uncorrelated_observables,
        )


@workflow_protocol()
class AverageObservable(BaseAverageObservable):
    """Computes the average value of an observable as well as bootstrapped
    uncertainties for the average.
    """

    observable = InputAttribute(
        docstring="The file path to the observable which should be averaged.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
    )
    divisor = InputAttribute(
        docstring="A value to divide the statistic by. This is useful if a statistic "
        "(such as enthalpy) needs to be normalised by the number of molecules.",
        type_hint=typing.Union[int, float, pint.Quantity],
        default_value=1.0,
    )

    def _observables(self) -> typing.Dict[str, ObservableArray]:
        return {"observable": self.observable / self.divisor}


@workflow_protocol()
class AverageDielectricConstant(BaseAverageObservable):
    """Computes the average value of the dielectric constant from a set of dipole
    moments (M) and volumes (V) sampled over the course of a molecular simulation such
    that ``eps = 1 + (<M^2> - <M>^2) / (3.0 * eps_0 * <V> * kb * T)`` [1]_.

    References
    ----------
    [1] A. Glattli, X. Daura and W. F. van Gunsteren. Derivation of an improved simple
        point charge model for liquid water: SPC/A and SPC/L. J. Chem. Phys. 116(22):
        9811-9828, 2002
    """

    dipole_moments = InputAttribute(
        docstring="The dipole moments of each sampled configuration.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
    )
    volumes = InputAttribute(
        docstring="The volume of each sampled configuration.",
        type_hint=ObservableArray,
        default_value=UNDEFINED,
    )

    def _observables(self):
        return {"dipole_moments": self.dipole_moments, "volumes": self.volumes}

    def _bootstrap_function(
        self,
        dipole_moments: ObservableArray,
        volumes: ObservableArray,
        **kwargs: ObservableArray,
    ):

        return compute_dielectric_constant(
            dipole_moments,
            volumes,
            self.thermodynamic_state.temperature,
            super(AverageDielectricConstant, self)._bootstrap_function,
        )


@workflow_protocol()
class AverageFreeEnergies(Protocol):
    """A protocol which computes the Boltzmann weighted average
    (ΔG° = -RT × Log[ Σ_{n} exp(-βΔG°_{n}) ]) of a set of free
    energies which were measured at the same thermodynamic state.
    Confidence intervals are computed by bootstrapping with replacement.
    """

    values: typing.List[Observable] = InputAttribute(
        docstring="The values to add together.", type_hint=list, default_value=UNDEFINED
    )
    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic state at which the free energies were measured.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    bootstrap_cycles = InputAttribute(
        docstring="The number of bootstrap cycles to perform when estimating "
        "the uncertainty in the combined free energies.",
        type_hint=int,
        default_value=2000,
    )

    result = OutputAttribute(docstring="The sum of the values.", type_hint=Observable)
    confidence_intervals = OutputAttribute(
        docstring="The 95% confidence intervals on the average free energy.",
        type_hint=pint.Quantity,
    )

    def _execute(self, directory, available_resources):

        from scipy.special import logsumexp

        default_unit = unit.kilocalorie / unit.mole

        boltzmann_factor = (
            self.thermodynamic_state.temperature * unit.molar_gas_constant
        )
        boltzmann_factor.ito(default_unit)

        beta = 1.0 / boltzmann_factor

        values = [
            (-beta * value.value.to(default_unit)).to(unit.dimensionless).magnitude
            for value in self.values
        ]

        # Compute the mean.
        mean = logsumexp(values)

        # Compute the gradients of the mean.
        value_gradients = [
            {gradient.key: -beta * gradient.value for gradient in value.gradients}
            for value in self.values
        ]
        value_gradients_by_key = {
            gradient_key: [
                gradients_by_key[gradient_key] for gradients_by_key in value_gradients
            ]
            for gradient_key in value_gradients[0]
        }

        mean_gradients = []

        for gradient_key, gradient_values in value_gradients_by_key.items():

            expected_unit = value_gradients[0][gradient_key].units

            d_log_mean_numerator, d_mean_numerator_sign = logsumexp(
                values,
                b=[x.to(expected_unit).magnitude for x in gradient_values],
                return_sign=True,
            )
            d_mean_numerator = d_mean_numerator_sign * np.exp(d_log_mean_numerator)

            d_mean_d_theta = d_mean_numerator / np.exp(mean)

            mean_gradients.append(
                ParameterGradient(
                    key=gradient_key,
                    value=-boltzmann_factor * d_mean_d_theta * expected_unit,
                )
            )

        # Compute the standard error and 95% CI
        cycle_result = np.empty(self.bootstrap_cycles)

        for cycle_index, cycle in enumerate(range(self.bootstrap_cycles)):

            cycle_values = np.empty(len(self.values))

            for value_index, value in enumerate(self.values):

                cycle_mean = value.value.to(default_unit).magnitude
                cycle_sem = value.error.to(default_unit).magnitude

                sampled_value = np.random.normal(cycle_mean, cycle_sem) * default_unit
                cycle_values[value_index] = (
                    (-beta * sampled_value).to(unit.dimensionless).magnitude
                )

            # ΔG° = -RT × Log[ Σ_{n} exp(-βΔG°_{n}) ]
            cycle_result[cycle_index] = logsumexp(cycle_values)

        mean = -boltzmann_factor * mean
        sem = np.std(-boltzmann_factor * cycle_result)

        confidence_intervals = np.empty(2)
        sorted_statistics = np.sort(cycle_result)

        confidence_intervals[0] = sorted_statistics[int(0.025 * self.bootstrap_cycles)]
        confidence_intervals[1] = sorted_statistics[int(0.975 * self.bootstrap_cycles)]

        confidence_intervals = -boltzmann_factor * confidence_intervals

        self.result = Observable(value=mean.plus_minus(sem), gradients=mean_gradients)
        self.confidence_intervals = confidence_intervals

    def validate(self, attribute_type=None):
        super(AverageFreeEnergies, self).validate(attribute_type)
        assert all(isinstance(x, Observable) for x in self.values)

        if len(self.values) == 0:
            return

        expected_gradients = {gradient.key for gradient in self.values[0].gradients}

        if not all(
            expected_gradients == {gradient.key for gradient in value.gradients}
            for value in self.values
        ):

            raise ValueError(
                "The values must contain gradient information for the same set of "
                "force field parameters."
            )


@workflow_protocol()
class ComputeDipoleMoments(Protocol):
    """A protocol which will compute the dipole moment for each configuration in
    a trajectory and for a given parameterized system."""

    parameterized_system = InputAttribute(
        docstring="The parameterized system which encodes the charge on each atom "
        "in the system.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )
    trajectory_path = InputAttribute(
        docstring="The file path to the trajectory of configurations.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    gradient_parameters = InputAttribute(
        docstring="An optional list of parameters to differentiate the dipole moments "
        "with respect to.",
        type_hint=list,
        default_value=lambda: list(),
    )

    dipole_moments = OutputAttribute(
        docstring="The computed dipole moments.", type_hint=ObservableArray
    )

    @classmethod
    def _extract_charges(
        cls, system: "openmm.System"
    ) -> typing.Optional[unit.Quantity]:
        """Retrieve the charge on each atom from an OpenMM system object.

        Parameters
        ----------
            The system object containing the charges.

        Returns
        -------
            The charge on each atom in the system if any are present, otherwise
            none.
        """
        from simtk import openmm
        from simtk import unit as simtk_unit

        forces = [
            system.getForce(force_index)
            for force_index in range(system.getNumForces())
            if isinstance(system.getForce(force_index), openmm.NonbondedForce)
        ]

        if len(forces) > 1:

            raise ValueError(
                f"The system must contain no more than one non-bonded force, however "
                f"{len(forces)} were found."
            )

        if len(forces) == 0:
            return None

        charges = np.array(
            [
                forces[0]
                .getParticleParameters(atom_index)[0]
                .value_in_unit(simtk_unit.elementary_charge)
                for atom_index in range(forces[0].getNumParticles())
            ]
        )
        return charges * unit.elementary_charge

    def _compute_charge_derivatives(self, n_atoms: int):

        d_charge_d_theta = {key: np.zeros(n_atoms) for key in self.gradient_parameters}

        if len(self.gradient_parameters) > 0 and not isinstance(
            self.parameterized_system.force_field, SmirnoffForceFieldSource
        ):
            raise ValueError(
                "Derivates can only be computed for systems parameterized with "
                "SMIRNOFF force fields."
            )

        force_field = self.parameterized_system.force_field.to_force_field()
        topology = self.parameterized_system.topology

        for key in self.gradient_parameters:

            reverse_system, reverse_value = system_subset(key, force_field, topology)
            forward_system, forward_value = system_subset(
                key, force_field, topology, 0.1
            )

            reverse_value = openmm_quantity_to_pint(reverse_value)
            forward_value = openmm_quantity_to_pint(forward_value)

            reverse_charges = self._extract_charges(reverse_system)
            forward_charges = self._extract_charges(forward_system)

            if reverse_charges is None and forward_charges is None:
                d_charge_d_theta[key] /= forward_value.units

            else:
                d_charge_d_theta[key] = (forward_charges - reverse_charges) / (
                    forward_value - reverse_value
                )

        return d_charge_d_theta

    def _execute(self, directory, available_resources):

        import mdtraj

        charges = self._extract_charges(self.parameterized_system.system)
        charge_derivatives = self._compute_charge_derivatives(len(charges))

        dipole_moments = []
        dipole_gradients = {key: [] for key in self.gradient_parameters}

        for chunk in mdtraj.iterload(
            self.trajectory_path, top=self.parameterized_system.topology_path, chunk=50
        ):

            xyz = chunk.xyz.transpose(0, 2, 1) * unit.nanometers

            dipole_moments.extend(xyz.dot(charges))

            for key in self.gradient_parameters:
                dipole_gradients[key].extend(xyz.dot(charge_derivatives[key]))

        self.dipole_moments = ObservableArray(
            value=np.vstack(dipole_moments),
            gradients=[
                ParameterGradient(key=key, value=np.vstack(dipole_gradients[key]))
                for key in self.gradient_parameters
            ],
        )


class BaseDecorrelateProtocol(Protocol, abc.ABC):
    """An abstract base class for protocols which will subsample
    a set of data, yielding only equilibrated, uncorrelated data.
    """

    time_series_statistics: typing.Union[
        TimeSeriesStatistics, typing.List[TimeSeriesStatistics]
    ] = InputAttribute(
        docstring="Statistics about the data to decorrelate. This should include the "
        "statistical inefficiency and the index after which the observables have "
        "become stationary (i.e. equilibrated). If a list of such statistics are "
        "provided it will be assumed that multiple time series which have been "
        "joined together are being decorrelated and hence will each be decorrelated "
        "separately.",
        type_hint=typing.Union[list, TimeSeriesStatistics],
        default_value=UNDEFINED,
    )

    def _n_expected(self) -> int:
        """Returns the expected number of samples to decorrelate."""

        time_series_statistics = self.time_series_statistics

        if isinstance(time_series_statistics, TimeSeriesStatistics):
            time_series_statistics = [time_series_statistics]

        return sum(statistics.n_total_points for statistics in time_series_statistics)

    def _uncorrelated_indices(self) -> typing.List[int]:
        """Returns the indices of the time series being decorrelated to retain."""

        time_series_statistics = self.time_series_statistics

        if isinstance(time_series_statistics, TimeSeriesStatistics):
            time_series_statistics = [time_series_statistics]

        n_cumulative = [
            0,
            *itertools.accumulate(
                [statistics.n_total_points for statistics in time_series_statistics]
            ),
        ]

        uncorrelated_indices = [
            n_cumulative[statistics_index] + index + statistics.equilibration_index
            for statistics_index, statistics in enumerate(time_series_statistics)
            for index in timeseries.get_uncorrelated_indices(
                statistics.n_total_points - statistics.equilibration_index,
                statistics.statistical_inefficiency,
            )
        ]

        assert len(uncorrelated_indices) == sum(
            statistics.n_uncorrelated_points for statistics in time_series_statistics
        )

        return uncorrelated_indices


@workflow_protocol()
class DecorrelateTrajectory(BaseDecorrelateProtocol):
    """A protocol which will subsample frames from a trajectory, yielding only
    uncorrelated frames as determined from a provided statistical inefficiency and
    equilibration time.
    """

    input_coordinate_file = InputAttribute(
        docstring="The file path to the starting coordinates of a trajectory.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    input_trajectory_path = InputAttribute(
        docstring="The file path to the trajectory to subsample.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    output_trajectory_path = OutputAttribute(
        docstring="The file path to the subsampled trajectory.", type_hint=str
    )

    @staticmethod
    def _yield_frame(file, topology, stride):
        """A generator which yields frames of a DCD trajectory.

        Parameters
        ----------
        file: mdtraj.DCDTrajectoryFile
            The file object being used to read the trajectory.
        topology: mdtraj.Topology
            The object which describes the topology of the trajectory.
        stride
            Only read every stride-th frame.

        Returns
        -------
        mdtraj.Trajectory
            A trajectory containing only a single frame.
        """

        while True:

            frame = file.read_as_traj(topology, n_frames=1, stride=stride)

            if len(frame) == 0:
                return

            yield frame

    def _execute(self, directory, available_resources):

        import mdtraj
        from mdtraj.formats.dcd import DCDTrajectoryFile
        from mdtraj.utils import in_units_of

        # Set the output path.
        self.output_trajectory_path = path.join(
            directory, "uncorrelated_trajectory.dcd"
        )

        # Load in the trajectories topology.
        topology = mdtraj.load_frame(self.input_coordinate_file, 0).topology
        # Parse the internal mdtraj distance unit. While private access is undesirable,
        # this is never publicly defined and I believe this route to be preferable
        # over hard coding this unit.
        # noinspection PyProtectedMember
        base_distance_unit = mdtraj.Trajectory._distance_unit

        # Determine the frames to retrain
        uncorrelated_indices = {*self._uncorrelated_indices()}

        frame_count = 0

        with DCDTrajectoryFile(self.input_trajectory_path, "r") as input_file:
            with DCDTrajectoryFile(self.output_trajectory_path, "w") as output_file:

                for frame in self._yield_frame(input_file, topology, 1):

                    if frame_count in uncorrelated_indices:

                        output_file.write(
                            xyz=in_units_of(
                                frame.xyz, base_distance_unit, output_file.distance_unit
                            ),
                            cell_lengths=in_units_of(
                                frame.unitcell_lengths,
                                base_distance_unit,
                                output_file.distance_unit,
                            ),
                            cell_angles=frame.unitcell_angles[0],
                        )

                    frame_count += 1

        assert frame_count == self._n_expected()


@workflow_protocol()
class DecorrelateObservables(BaseDecorrelateProtocol):
    """A protocol which will subsample a trajectory of observables, yielding only
    uncorrelated entries as determined from a provided statistical inefficiency and
    equilibration time.
    """

    input_observables = InputAttribute(
        docstring="The observables to decorrelate.",
        type_hint=typing.Union[ObservableArray, ObservableFrame],
        default_value=UNDEFINED,
    )

    output_observables = OutputAttribute(
        docstring="The decorrelated observables.",
        type_hint=typing.Union[ObservableArray, ObservableFrame],
    )

    def _execute(self, directory, available_resources):

        assert len(self.input_observables) == self._n_expected()

        uncorrelated_indices = self._uncorrelated_indices()
        uncorrelated_observable = self.input_observables.subset(uncorrelated_indices)

        self.output_observables = uncorrelated_observable
