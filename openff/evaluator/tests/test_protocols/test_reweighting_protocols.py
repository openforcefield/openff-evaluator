import tempfile

import numpy as np
import pytest
from openff.units import unit

from openff.evaluator.backends import ComputeResources
from openff.evaluator.protocols.reweighting import (
    ConcatenateObservables,
    ConcatenateTrajectories,
    ReweightDielectricConstant,
    ReweightObservable,
)
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import ObservableArray, ObservableFrame


def test_concatenate_trajectories():

    import mdtraj

    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    original_trajectory = mdtraj.load(trajectory_path, top=coordinate_path)

    with tempfile.TemporaryDirectory() as temporary_directory:

        concatenate_protocol = ConcatenateTrajectories("concatenate_protocol")
        concatenate_protocol.input_coordinate_paths = [coordinate_path, coordinate_path]
        concatenate_protocol.input_trajectory_paths = [trajectory_path, trajectory_path]
        concatenate_protocol.execute(temporary_directory, ComputeResources())

        final_trajectory = mdtraj.load(
            concatenate_protocol.output_trajectory_path, top=coordinate_path
        )
        assert len(final_trajectory) == len(original_trajectory) * 2


@pytest.mark.parametrize(
    "observables",
    [
        [ObservableArray(value=np.zeros((2, 3)) * unit.kelvin)],
        [ObservableArray(value=np.zeros((2, 3)) * unit.kelvin)] * 2,
        [
            ObservableFrame(
                {"Temperature": ObservableArray(value=np.zeros((2, 3)) * unit.kelvin)}
            )
        ],
        [
            ObservableFrame(
                {"Temperature": ObservableArray(value=np.zeros((2, 3)) * unit.kelvin)}
            )
        ]
        * 2,
    ],
)
def test_concatenate_observables(observables):

    concatenate_protocol = ConcatenateObservables("")
    concatenate_protocol.input_observables = observables
    concatenate_protocol.execute()

    assert len(concatenate_protocol.output_observables) == 2 * len(observables)


def test_reweight_observables():

    with tempfile.TemporaryDirectory() as directory:

        reweight_protocol = ReweightObservable("")
        reweight_protocol.observable = ObservableArray(value=np.zeros(10) * unit.kelvin)
        reweight_protocol.reference_reduced_potentials = [
            ObservableArray(value=np.zeros(10) * unit.dimensionless)
        ]
        reweight_protocol.frame_counts = [10]
        reweight_protocol.target_reduced_potentials = ObservableArray(
            value=np.zeros(10) * unit.dimensionless
        )
        reweight_protocol.bootstrap_uncertainties = True
        reweight_protocol.required_effective_samples = 0
        reweight_protocol.execute(directory, ComputeResources())


def test_reweight_dielectric_constant():

    with tempfile.TemporaryDirectory() as directory:

        reweight_protocol = ReweightDielectricConstant("")
        reweight_protocol.dipole_moments = ObservableArray(
            value=np.zeros((10, 3)) * unit.elementary_charge * unit.nanometers
        )
        reweight_protocol.volumes = ObservableArray(
            value=np.ones((10, 1)) * unit.nanometer**3
        )
        reweight_protocol.reference_reduced_potentials = [
            ObservableArray(value=np.zeros(10) * unit.dimensionless)
        ]
        reweight_protocol.target_reduced_potentials = ObservableArray(
            value=np.zeros(10) * unit.dimensionless
        )
        reweight_protocol.thermodynamic_state = ThermodynamicState(
            298.15 * unit.kelvin, 1.0 * unit.atmosphere
        )
        reweight_protocol.frame_counts = [10]
        reweight_protocol.bootstrap_uncertainties = True
        reweight_protocol.required_effective_samples = 0
        reweight_protocol.execute(directory, ComputeResources())
