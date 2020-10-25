import inspect
from random import randint, random

import mdtraj
import numpy as np
import pytest
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField, vdWHandler
from openforcefield.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
)
from simtk import unit as simtk_unit

from openff.evaluator import unit
from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.protocols.openmm import _compute_gradients
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import ObservableArray, ObservableFrame
from openff.evaluator.utils.openmm import (
    openmm_quantity_to_pint,
    openmm_unit_to_pint,
    pint_quantity_to_openmm,
    pint_unit_to_openmm,
    system_subset,
    unsupported_openmm_units,
)


def _get_all_openmm_units():
    """Returns a list of all of the defined OpenMM units.

    Returns
    -------
    list of simtk.unit.Unit
    """

    all_openmm_units = set(
        [
            unit_type
            for _, unit_type in inspect.getmembers(simtk_unit)
            if isinstance(unit_type, simtk_unit.Unit)
        ]
    )

    all_openmm_units = all_openmm_units.difference(unsupported_openmm_units)

    return all_openmm_units


def _get_all_pint_units():
    """Returns a list of all of the pint units which
    could be converted from all OpenMM units.

    Returns
    -------
    list of pint.Unit
    """

    all_openmm_units = _get_all_openmm_units()

    all_pint_units = [
        openmm_unit_to_pint(openmm_unit) for openmm_unit in all_openmm_units
    ]

    return all_pint_units


def test_daltons():

    openmm_quantity = random() * simtk_unit.dalton
    openmm_raw_value = openmm_quantity.value_in_unit(simtk_unit.gram / simtk_unit.mole)

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.to(unit.gram / unit.mole).magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("openmm_unit", _get_all_openmm_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_openmm_unit_to_pint(openmm_unit, value):

    openmm_quantity = value * openmm_unit
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("pint_unit", _get_all_pint_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_pint_to_openmm(pint_unit, value):

    pint_quantity = value * pint_unit
    pint_raw_value = pint_quantity.magnitude

    openmm_quantity = pint_quantity_to_openmm(pint_quantity)
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_quantity.unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


def test_combinatorial_pint_to_openmm():

    all_pint_units = _get_all_pint_units()

    for i in range(len(all_pint_units)):

        for j in range(i, len(all_pint_units)):

            pint_unit = all_pint_units[i] * all_pint_units[j]

            pint_quantity = random() * pint_unit
            pint_raw_value = pint_quantity.magnitude

            openmm_quantity = pint_quantity_to_openmm(pint_quantity)
            openmm_raw_value = openmm_quantity.value_in_unit(openmm_quantity.unit)

            assert np.isclose(openmm_raw_value, pint_raw_value)


def test_combinatorial_openmm_to_pint():

    all_openmm_units = list(_get_all_openmm_units())

    for i in range(len(all_openmm_units)):

        for j in range(i, len(all_openmm_units)):

            openmm_unit = all_openmm_units[i] * all_openmm_units[j]

            openmm_quantity = random() * openmm_unit
            openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

            pint_quantity = openmm_quantity_to_pint(openmm_quantity)
            pint_raw_value = pint_quantity.magnitude

            assert np.isclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize(
    "openmm_unit",
    {*_get_all_openmm_units(), simtk_unit.dalton ** 2, simtk_unit.dalton ** 3},
)
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_openmm_unit_conversions(openmm_unit, value):

    openmm_quantity = value * openmm_unit

    openmm_base_quantity = openmm_quantity.in_unit_system(simtk_unit.md_unit_system)

    if not isinstance(openmm_base_quantity, simtk_unit.Quantity):
        openmm_base_quantity *= simtk_unit.dimensionless

    pint_base_quantity = openmm_quantity_to_pint(openmm_base_quantity)

    pint_unit = openmm_unit_to_pint(openmm_unit)
    pint_quantity = pint_base_quantity.to(pint_unit)

    pint_raw_value = pint_quantity.magnitude
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("pint_unit", _get_all_pint_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_pint_unit_conversions(pint_unit, value):

    pint_quantity = value * pint_unit

    pint_base_quantity = pint_quantity.to_base_units()
    openmm_base_quantity = pint_quantity_to_openmm(pint_base_quantity)

    openmm_unit = pint_unit_to_openmm(pint_unit)
    openmm_quantity = openmm_base_quantity.in_units_of(openmm_unit)

    pint_raw_value = pint_quantity.magnitude

    if pint_unit == unit.dimensionless and (
        isinstance(openmm_quantity, float)
        or isinstance(openmm_quantity, int)
        or isinstance(openmm_quantity, list)
        or isinstance(openmm_quantity, np.ndarray)
    ):
        openmm_raw_value = openmm_quantity
    else:
        openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


def test_constants():

    assert np.isclose(
        simtk_unit.AVOGADRO_CONSTANT_NA.value_in_unit((1.0 / simtk_unit.mole).unit),
        (1.0 * unit.avogadro_constant).to((1.0 / unit.mole).units).magnitude,
    )

    assert np.isclose(
        simtk_unit.BOLTZMANN_CONSTANT_kB.value_in_unit(
            simtk_unit.joule / simtk_unit.kelvin
        ),
        (1.0 * unit.boltzmann_constant).to(unit.joule / unit.kelvin).magnitude,
    )

    assert np.isclose(
        simtk_unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            simtk_unit.joule / simtk_unit.kelvin / simtk_unit.mole
        ),
        (1.0 * unit.molar_gas_constant)
        .to(unit.joule / unit.kelvin / unit.mole)
        .magnitude,
    )

    assert np.isclose(
        simtk_unit.GRAVITATIONAL_CONSTANT_G.value_in_unit(
            simtk_unit.meter ** 2 * simtk_unit.newton / simtk_unit.kilogram ** 2
        ),
        (1.0 * unit.newtonian_constant_of_gravitation)
        .to(unit.meter ** 2 * unit.newton / unit.kilogram ** 2)
        .magnitude,
    )

    assert np.isclose(
        simtk_unit.SPEED_OF_LIGHT_C.value_in_unit(
            simtk_unit.meter / simtk_unit.seconds
        ),
        (1.0 * unit.speed_of_light).to(unit.meter / unit.seconds).magnitude,
    )


def hydrogen_chloride_force_field(
    library_charge: bool, charge_increment: bool
) -> ForceField:
    """Returns a SMIRNOFF force field which is able to parameterize hydrogen chloride."""

    # Create the FF
    force_field = ForceField()

    # Add a Vdw handler.
    vdw_handler = vdWHandler(version=0.3)
    vdw_handler.method = "cutoff"
    vdw_handler.cutoff = 6.0 * simtk_unit.angstrom
    vdw_handler.scale14 = 1.0
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "epsilon": 0.0 * simtk_unit.kilojoules_per_mole,
            "sigma": 1.0 * simtk_unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#17:1]",
            "epsilon": 2.0 * simtk_unit.kilojoules_per_mole,
            "sigma": 2.0 * simtk_unit.angstrom,
        }
    )
    force_field.register_parameter_handler(vdw_handler)

    # Add an electrostatic, a library charge and a charge increment handler.
    electrostatics_handler = ElectrostaticsHandler(version=0.3)
    electrostatics_handler.cutoff = 6.0 * simtk_unit.angstrom
    electrostatics_handler.method = "PME"
    force_field.register_parameter_handler(electrostatics_handler)

    if library_charge:

        library_charge_handler = LibraryChargeHandler(version=0.3)
        library_charge_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#1:1]",
                "charge1": 1.0 * simtk_unit.elementary_charge,
            }
        )
        library_charge_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#17:1]",
                "charge1": -1.0 * simtk_unit.elementary_charge,
            }
        )
        force_field.register_parameter_handler(library_charge_handler)

    if charge_increment:

        charge_increment_handler = ChargeIncrementModelHandler(version=0.3)
        charge_increment_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#1:1]-[#17:2]",
                "charge_increment1": -1.0 * simtk_unit.elementary_charge,
                "charge_increment2": 1.0 * simtk_unit.elementary_charge,
            }
        )
        force_field.register_parameter_handler(charge_increment_handler)

    return force_field


def test_system_subset_vdw():

    # Create a dummy topology
    topology = Molecule.from_smiles("Cl").to_topology()

    # Create the system subset.
    system, parameter_value = system_subset(
        parameter_key=ParameterGradientKey("vdW", "[#1:1]", "epsilon"),
        force_field=hydrogen_chloride_force_field(True, True),
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    charge_0, sigma_0, epsilon_0 = system.getForce(0).getParticleParameters(0)
    charge_1, sigma_1, epsilon_1 = system.getForce(0).getParticleParameters(1)

    assert np.isclose(charge_0.value_in_unit(simtk_unit.elementary_charge), 0.0)
    assert np.isclose(charge_1.value_in_unit(simtk_unit.elementary_charge), 0.0)

    assert np.isclose(sigma_0.value_in_unit(simtk_unit.angstrom), 2.0)
    assert np.isclose(sigma_1.value_in_unit(simtk_unit.angstrom), 1.0)

    assert np.isclose(epsilon_0.value_in_unit(simtk_unit.kilojoules_per_mole), 2.0)
    assert np.isclose(epsilon_1.value_in_unit(simtk_unit.kilojoules_per_mole), 0.5)


def test_system_subset_library_charge():

    force_field = hydrogen_chloride_force_field(True, False)

    # Ensure a zero charge after perturbation.
    force_field.get_parameter_handler("LibraryCharges").parameters["[#1:1]"].charge1 = (
        1.5 * simtk_unit.elementary_charge
    )

    # Create a dummy topology
    topology = Molecule.from_smiles("Cl").to_topology()

    # Create the system subset.
    system, parameter_value = system_subset(
        parameter_key=ParameterGradientKey("LibraryCharges", "[#17:1]", "charge1"),
        force_field=force_field,
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    charge_0, sigma_0, epsilon_0 = system.getForce(0).getParticleParameters(0)
    charge_1, sigma_1, epsilon_1 = system.getForce(0).getParticleParameters(1)

    assert np.isclose(charge_0.value_in_unit(simtk_unit.elementary_charge), -1.5)
    assert np.isclose(charge_1.value_in_unit(simtk_unit.elementary_charge), 1.5)

    assert np.isclose(sigma_0.value_in_unit(simtk_unit.angstrom), 10.0)
    assert np.isclose(sigma_1.value_in_unit(simtk_unit.angstrom), 10.0)

    assert np.isclose(epsilon_0.value_in_unit(simtk_unit.kilojoules_per_mole), 0.0)
    assert np.isclose(epsilon_1.value_in_unit(simtk_unit.kilojoules_per_mole), 0.0)


def test_system_subset_charge_increment():

    pytest.skip(
        "This test will fail until the SMIRNOFF charge increment handler allows "
        "N - 1 charges to be specified."
    )

    # Create a dummy topology
    topology = Molecule.from_smiles("Cl").to_topology()

    # Create the system subset.
    system, parameter_value = system_subset(
        parameter_key=ParameterGradientKey(
            "ChargeIncrementModel", "[#1:1]-[#17:2]", "charge_increment1"
        ),
        force_field=hydrogen_chloride_force_field(False, True),
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    charge_0, sigma_0, epsilon_0 = system.getForce(0).getParticleParameters(0)
    charge_1, sigma_1, epsilon_1 = system.getForce(0).getParticleParameters(1)

    assert not np.isclose(charge_0.value_in_unit(simtk_unit.elementary_charge), -1.0)
    assert np.isclose(charge_1.value_in_unit(simtk_unit.elementary_charge), 1.0)

    assert np.isclose(sigma_0.value_in_unit(simtk_unit.angstrom), 10.0)
    assert np.isclose(sigma_1.value_in_unit(simtk_unit.angstrom), 10.0)

    assert np.isclose(epsilon_0.value_in_unit(simtk_unit.kilojoules_per_mole), 0.0)
    assert np.isclose(epsilon_1.value_in_unit(simtk_unit.kilojoules_per_mole), 0.0)


@pytest.mark.parametrize("smirks, all_zeros", [("[#6X4:1]", True), ("[#8:1]", False)])
def test_compute_gradients(tmpdir, smirks, all_zeros):

    # Load a short trajectory.
    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    trajectory = mdtraj.load_dcd(trajectory_path, coordinate_path)

    observables = ObservableFrame(
        {
            "PotentialEnergy": ObservableArray(
                np.zeros(len(trajectory)) * unit.kilojoule / unit.mole
            )
        }
    )

    _compute_gradients(
        [ParameterGradientKey("vdW", smirks, "epsilon")],
        observables,
        ForceField("openff-1.2.0.offxml"),
        ThermodynamicState(298.15 * unit.kelvin, 1.0 * unit.atmosphere),
        Topology.from_mdtraj(trajectory.topology, [Molecule.from_smiles("O")]),
        trajectory,
        ComputeResources(),
        True,
    )

    assert len(observables["PotentialEnergy"].gradients[0].value) == len(trajectory)

    if all_zeros:
        assert np.allclose(
            observables["PotentialEnergy"].gradients[0].value,
            0.0 * unit.kilojoule / unit.kilocalorie,
        )
    else:
        assert not np.allclose(
            observables["PotentialEnergy"].gradients[0].value,
            0.0 * unit.kilojoule / unit.kilocalorie,
        )
