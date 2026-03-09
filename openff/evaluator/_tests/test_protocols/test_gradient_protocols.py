import os
import tempfile

import numpy as np
from openff.toolkit.typing.engines.smirnoff import ForceField, vdWHandler
from openff.toolkit.typing.engines.smirnoff.parameters import (
    BondHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    VirtualSiteHandler,
)
from openff.units import unit

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

    force_field = ForceField()

    vdw_handler = vdWHandler(version=0.4)
    vdw_handler.cutoff = 6.0 * unit.angstrom
    vdw_handler.scale14 = 1.0
    for smirks, eps in (("[#1:1]", 0.0), ("[#17:1]", 2.0)):
        vdw_handler.add_parameter(
            {
                "smirks": smirks,
                "epsilon": eps * unit.kilojoules_per_mole,
                "sigma": 1.0 * unit.angstrom,
            }
        )
    force_field.register_parameter_handler(vdw_handler)

    bond_handler = BondHandler(version=0.4)
    bond_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "length": 1.0 * unit.angstrom,
            "k": 1000.0 * unit.kilojoule_per_mole / unit.angstrom**2,
        }
    )
    force_field.register_parameter_handler(bond_handler)

    electrostatics_handler = ElectrostaticsHandler(version=0.3)
    electrostatics_handler.cutoff = 6.0 * unit.angstrom
    electrostatics_handler.periodic_potential = "PME"
    force_field.register_parameter_handler(electrostatics_handler)

    library_charge_handler = LibraryChargeHandler(version=0.3)
    library_charge_handler.add_parameter(
        {"smirks": "[#1:1]", "charge1": 1.0 * unit.elementary_charge}
    )
    library_charge_handler.add_parameter(
        {"smirks": "[#17:1]", "charge1": -1.0 * unit.elementary_charge}
    )
    force_field.register_parameter_handler(library_charge_handler)

    vsite_handler = VirtualSiteHandler(version=0.3)
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "type": "BondCharge",
            "distance": 0.10 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
        }
    )
    force_field.register_parameter_handler(vsite_handler)

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
