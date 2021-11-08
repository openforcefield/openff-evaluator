import os
import tempfile

import numpy as np
import pytest
from openff.units import unit

from openff.evaluator.forcefield import ParameterGradient, ParameterGradientKey
from openff.evaluator.protocols.analysis import (
    AverageDielectricConstant,
    AverageFreeEnergies,
    AverageObservable,
    ComputeDipoleMoments,
    DecorrelateObservables,
    DecorrelateTrajectory,
)
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.substances import Substance
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import Observable, ObservableArray
from openff.evaluator.utils.timeseries import TimeSeriesStatistics


def test_average_observable():

    with tempfile.TemporaryDirectory() as temporary_directory:

        average_observable = AverageObservable("")
        average_observable.observable = ObservableArray(1.0 * unit.kelvin)
        average_observable.bootstrap_iterations = 1
        average_observable.execute(temporary_directory)

        assert np.isclose(average_observable.value.value, 1.0 * unit.kelvin)


def test_average_dielectric_constant():

    with tempfile.TemporaryDirectory() as temporary_directory:

        average_observable = AverageDielectricConstant("")
        average_observable.dipole_moments = ObservableArray(
            np.zeros((1, 3)) * unit.elementary_charge * unit.nanometer
        )
        average_observable.volumes = ObservableArray(
            np.ones((1, 1)) * unit.nanometer ** 3
        )
        average_observable.thermodynamic_state = ThermodynamicState(
            298.15 * unit.kelvin, 1.0 * unit.atmosphere
        )
        average_observable.bootstrap_iterations = 1
        average_observable.execute(temporary_directory)

        assert np.isclose(average_observable.value.value, 1.0 * unit.dimensionless)


def test_average_free_energies_protocol():
    """Tests adding together two free energies."""

    delta_g_one = Observable(
        value=(-10.0 * unit.kilocalorie / unit.mole).plus_minus(
            1.0 * unit.kilocalorie / unit.mole
        ),
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "sigma"),
                value=0.1 * unit.kilocalorie / unit.mole / unit.angstrom,
            )
        ],
    )
    delta_g_two = Observable(
        value=(-20.0 * unit.kilocalorie / unit.mole).plus_minus(
            2.0 * unit.kilocalorie / unit.mole
        ),
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "sigma"),
                value=0.2 * unit.kilocalorie / unit.mole / unit.angstrom,
            )
        ],
    )

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1 * unit.atmosphere)

    sum_protocol = AverageFreeEnergies("")

    sum_protocol.values = [delta_g_one, delta_g_two]
    sum_protocol.thermodynamic_state = thermodynamic_state

    sum_protocol.execute()

    result_value = sum_protocol.result.value.to(unit.kilocalorie / unit.mole)
    result_uncertainty = sum_protocol.result.error.to(unit.kilocalorie / unit.mole)

    assert isinstance(sum_protocol.result, Observable)
    assert result_value.magnitude == pytest.approx(-20.0, abs=0.2)
    assert result_uncertainty.magnitude == pytest.approx(2.0, abs=0.2)

    assert (
        sum_protocol.confidence_intervals[0]
        > result_value
        > sum_protocol.confidence_intervals[1]
    )

    gradient_value = sum_protocol.result.gradients[0].value.to(
        unit.kilocalorie / unit.mole / unit.angstrom
    )
    beta = 1.0 / (298.0 * unit.kelvin * unit.molar_gas_constant).to(
        unit.kilocalorie / unit.mole
    )

    assert np.isclose(
        gradient_value.magnitude,
        (0.1 * np.exp(-beta.magnitude * -10.0) + 0.2 * np.exp(-beta.magnitude * -20.0))
        / (np.exp(-beta.magnitude * -10.0) + np.exp(-beta.magnitude * -20.0)),
    )


def test_compute_dipole_moments(tmpdir):

    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    # Build a system object for water
    force_field_path = os.path.join(tmpdir, "ff.json")

    with open(force_field_path, "w") as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    assign_parameters = BuildSmirnoffSystem("")
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = coordinate_path
    assign_parameters.substance = Substance.from_components("O")
    assign_parameters.execute(str(tmpdir))

    # TODO - test gradients when TIP3P library charges added.
    protocol = ComputeDipoleMoments("")
    protocol.parameterized_system = assign_parameters.parameterized_system
    protocol.trajectory_path = trajectory_path
    protocol.execute(str(tmpdir))

    assert len(protocol.dipole_moments) == 10
    assert protocol.dipole_moments.value.shape[1] == 3

    assert not np.allclose(
        protocol.dipole_moments.value, 0.0 * unit.elementary_charge * unit.nanometers
    )


def test_decorrelate_trajectory():

    import mdtraj

    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    with tempfile.TemporaryDirectory() as temporary_directory:

        protocol = DecorrelateTrajectory("")
        protocol.input_coordinate_file = coordinate_path
        protocol.input_trajectory_path = trajectory_path
        protocol.time_series_statistics = TimeSeriesStatistics(10, 4, 2.0, 2)
        protocol.execute(temporary_directory)

        final_trajectory = mdtraj.load(
            protocol.output_trajectory_path, top=coordinate_path
        )
        assert len(final_trajectory) == 4


def test_decorrelate_observables():

    with tempfile.TemporaryDirectory() as temporary_directory:

        protocol = DecorrelateObservables("")
        protocol.input_observables = ObservableArray(
            np.ones((10, 1)) * unit.nanometer ** 3
        )
        protocol.time_series_statistics = TimeSeriesStatistics(10, 4, 2.0, 2)
        protocol.execute(temporary_directory)

        assert len(protocol.output_observables) == 4
