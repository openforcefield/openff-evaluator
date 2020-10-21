import os
import tempfile

import numpy as np

from openff.evaluator import unit
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.protocols.gradients import ZeroGradients
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.utils.observables import ObservableArray


def test_zero_gradient():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = os.path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        gradient_key = ParameterGradientKey("vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon")

        zero_gradients = ZeroGradients("")
        zero_gradients.input_observables = ObservableArray(value=0.0 * unit.kelvin)
        zero_gradients.gradient_parameters = [gradient_key]
        zero_gradients.force_field_path = force_field_path
        zero_gradients.execute()

        assert len(zero_gradients.output_observables.gradients) == 1
        assert zero_gradients.output_observables.gradients[0].key == gradient_key
        assert np.allclose(zero_gradients.output_observables.gradients[0].value, 0.0)
