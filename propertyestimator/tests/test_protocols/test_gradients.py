import random
import tempfile
from os import path

import pytest

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.forcefield import ParameterGradientKey
from propertyestimator.protocols.gradients import (
    CentralDifferenceGradient,
    GradientReducedPotentials,
)
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.exceptions import PropertyEstimatorException


@pytest.mark.parametrize("use_subset", [True, False])
def test_gradient_reduced_potentials(use_subset):

    substance = Substance.from_components("O")
    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        reduced_potentials = GradientReducedPotentials(f"reduced_potentials")
        reduced_potentials.substance = substance
        reduced_potentials.thermodynamic_state = thermodynamic_state
        reduced_potentials.statistics_path = get_data_filename(
            "test/statistics/stats_pandas.csv"
        )
        reduced_potentials.force_field_path = force_field_path
        reduced_potentials.trajectory_file_path = get_data_filename(
            "test/trajectories/water.dcd"
        )
        reduced_potentials.coordinate_file_path = get_data_filename(
            "test/trajectories/water.pdb"
        )
        reduced_potentials.use_subset_of_force_field = use_subset
        reduced_potentials.enable_pbc = True
        reduced_potentials.parameter_key = ParameterGradientKey(
            "vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon"
        )

        result = reduced_potentials.execute(directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        assert path.isfile(reduced_potentials.forward_potentials_path)
        assert path.isfile(reduced_potentials.reverse_potentials_path)


def test_central_difference_gradient():

    with tempfile.TemporaryDirectory() as directory:

        gradient_key = ParameterGradientKey("vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon")

        reverse_parameter = -random.random() * unit.kelvin
        reverse_observable = -random.random() * unit.kelvin

        forward_parameter = random.random() * unit.kelvin
        forward_observable = random.random() * unit.kelvin

        central_difference = CentralDifferenceGradient("central_difference")
        central_difference.parameter_key = gradient_key
        central_difference.reverse_observable_value = reverse_observable
        central_difference.reverse_parameter_value = reverse_parameter
        central_difference.forward_observable_value = forward_observable
        central_difference.forward_parameter_value = forward_parameter

        result = central_difference.execute(directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)

        assert central_difference.gradient.value == (
            (forward_observable - reverse_observable)
            / (forward_parameter - reverse_parameter)
        )
