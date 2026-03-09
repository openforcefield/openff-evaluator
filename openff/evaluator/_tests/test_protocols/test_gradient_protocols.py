import os
import tempfile

import numpy as np
from openff.units import unit

from openff.evaluator._tests.test_utils.test_openmm import hydrogen_chloride_force_field
from openff.evaluator._tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.forcefield import ParameterGradientKey, SmirnoffForceFieldSource
from openff.evaluator.protocols.gradients import ZeroGradients
from openff.evaluator.utils.observables import ObservableArray


def test_zero_gradient():
    with tempfile.TemporaryDirectory() as directory:
        force_field_path = os.path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        gradient_keys = [
            ParameterGradientKey("vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon"),
            ParameterGradientKey("vdW", None, "scale14"),
        ]

        zero_gradients = ZeroGradients("")
        zero_gradients.input_observables = ObservableArray(value=0.0 * unit.kelvin)
        zero_gradients.gradient_parameters = gradient_keys
        zero_gradients.force_field_path = force_field_path
        zero_gradients.execute()

        assert len(zero_gradients.output_observables.gradients) == 2

        assert {
            gradient.key for gradient in zero_gradients.output_observables.gradients
        } == {*gradient_keys}

        for gradient in zero_gradients.output_observables.gradients:
            assert np.allclose(gradient.value, 0.0)


def test_zero_gradient_vsite():
    """ZeroGradients._execute must correctly resolve the parameter unit for a
    vsite gradient key, going through get_parameter_from_gradient_key with
    full identity fields (type/name/match)."""

    force_field = hydrogen_chloride_force_field(
        library_charge=True,
        charge_increment=False,
        vsite=True,
    )

    vsite_key = ParameterGradientKey(
        "VirtualSites",
        "[#1:1]-[#17:2]",
        "distance",
        virtual_site_type="BondCharge",
        virtual_site_name="EP",
        virtual_site_match="all_permutations",
    )

    with tempfile.TemporaryDirectory() as directory:
        force_field_path = os.path.join(directory, "ff.json")
        with open(force_field_path, "w") as file:
            file.write(SmirnoffForceFieldSource.from_object(force_field).json())

        zero_gradients = ZeroGradients("")
        zero_gradients.input_observables = ObservableArray(value=0.0 * unit.kelvin)
        zero_gradients.gradient_parameters = [vsite_key]
        zero_gradients.force_field_path = force_field_path
        zero_gradients.execute()

    assert len(zero_gradients.output_observables.gradients) == 1
    gradient = zero_gradients.output_observables.gradients[0]
    assert gradient.key == vsite_key
    assert np.allclose(gradient.value.m, 0.0)
