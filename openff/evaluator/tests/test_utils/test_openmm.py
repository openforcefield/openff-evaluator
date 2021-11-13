import os.path
from random import randint, random

import mdtraj
import numpy
import numpy as np

try:
    import openmm
except ImportError:
    from simtk import openmm

import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField, vdWHandler
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ChargeIncrementModelHandler,
    ElectrostaticsHandler,
    LibraryChargeHandler,
    VirtualSiteHandler,
)

try:
    from openmm import unit as openmm_unit
    from openmm.app import PDBFile
except ImportError:
    from simtk.openmm import unit as openmm_unit
    from simtk.openmm.app import PDBFile

from openff.evaluator import unit
from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.protocols.openmm import _compute_gradients
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import ObservableArray, ObservableFrame
from openff.evaluator.utils.openmm import (
    extract_atom_indices,
    extract_positions,
    openmm_quantity_to_pint,
    pint_quantity_to_openmm,
    system_subset,
    update_context_with_pdb,
    update_context_with_positions,
)


def test_daltons():

    openmm_quantity = random() * openmm_unit.dalton
    openmm_raw_value = openmm_quantity.value_in_unit(
        openmm_unit.gram / openmm_unit.mole
    )

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.to(unit.gram / unit.mole).magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize(
    "openmm_unit",
    [
        openmm_unit.dalton,
        openmm_unit.kilojoules_per_mole,
        openmm_unit.angstrom,
        openmm_unit.kelvin,
        openmm_unit.atmosphere,
        openmm_unit.gram,
        openmm_unit.liter,
        openmm_unit.gram / openmm_unit.liter,
    ],
)
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_openmm_to_pint(openmm_unit, value):

    openmm_quantity = value * openmm_unit
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize(
    "pint_unit",
    [
        unit.dalton,
        unit.kilojoules / unit.mole,
        unit.angstrom,
        unit.kelvin,
        unit.atmosphere,
        unit.gram,
        unit.liter,
        unit.gram / unit.liter,
    ],
)
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


def test_constants():

    assert np.isclose(
        openmm_unit.AVOGADRO_CONSTANT_NA.value_in_unit((1.0 / openmm_unit.mole).unit),
        (1.0 * unit.avogadro_constant).to((1.0 / unit.mole).units).magnitude,
    )

    assert np.isclose(
        openmm_unit.BOLTZMANN_CONSTANT_kB.value_in_unit(
            openmm_unit.joule / openmm_unit.kelvin
        ),
        (1.0 * unit.boltzmann_constant).to(unit.joule / unit.kelvin).magnitude,
    )

    assert np.isclose(
        openmm_unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            openmm_unit.joule / openmm_unit.kelvin / openmm_unit.mole
        ),
        (1.0 * unit.molar_gas_constant)
        .to(unit.joule / unit.kelvin / unit.mole)
        .magnitude,
    )

    assert np.isclose(
        openmm_unit.SPEED_OF_LIGHT_C.value_in_unit(
            openmm_unit.meter / openmm_unit.seconds
        ),
        (1.0 * unit.speed_of_light).to(unit.meter / unit.seconds).magnitude,
    )


def hydrogen_chloride_force_field(
    library_charge: bool,
    charge_increment: bool,
    vsite: bool,
) -> ForceField:
    """Returns a SMIRNOFF force field which is able to parameterize hydrogen chloride."""

    # Create the FF
    force_field = ForceField()

    # Add a Vdw handler.
    vdw_handler = vdWHandler(version=0.3)
    vdw_handler.method = "cutoff"
    vdw_handler.cutoff = 6.0 * openmm_unit.angstrom
    vdw_handler.scale14 = 1.0
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "epsilon": 0.0 * openmm_unit.kilojoules_per_mole,
            "sigma": 1.0 * openmm_unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#17:1]",
            "epsilon": 2.0 * openmm_unit.kilojoules_per_mole,
            "sigma": 2.0 * openmm_unit.angstrom,
        }
    )
    force_field.register_parameter_handler(vdw_handler)

    # Add an electrostatic, a library charge and a charge increment handler.
    electrostatics_handler = ElectrostaticsHandler(version=0.3)
    electrostatics_handler.cutoff = 6.0 * openmm_unit.angstrom
    electrostatics_handler.method = "PME"
    force_field.register_parameter_handler(electrostatics_handler)

    if library_charge:

        library_charge_handler = LibraryChargeHandler(version=0.3)
        library_charge_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#1:1]",
                "charge1": 1.0 * openmm_unit.elementary_charge,
            }
        )
        library_charge_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#17:1]",
                "charge1": -1.0 * openmm_unit.elementary_charge,
            }
        )
        force_field.register_parameter_handler(library_charge_handler)

    if charge_increment:

        charge_increment_handler = ChargeIncrementModelHandler(version=0.3)
        charge_increment_handler.add_parameter(
            parameter_kwargs={
                "smirks": "[#1:1]-[#17:2]",
                "charge_increment1": -1.0 * openmm_unit.elementary_charge,
                "charge_increment2": 1.0 * openmm_unit.elementary_charge,
            }
        )
        force_field.register_parameter_handler(charge_increment_handler)

    if vsite:

        vsite_handler = VirtualSiteHandler(version=0.3)
        vsite_handler.add_parameter(
            {
                "smirks": "[#1:1]-[#17:2]",
                "type": "BondCharge",
                "distance": -0.2 * openmm_unit.nanometers,
                "match": "once",
                "charge_increment1": 0.0 * openmm_unit.elementary_charge,
                "charge_increment2": 0.0 * openmm_unit.elementary_charge,
            }
        )
        force_field.register_parameter_handler(vsite_handler)

    return force_field


def test_system_subset_vdw():

    # Create a dummy topology
    topology = Molecule.from_smiles("Cl").to_topology()

    # Create the system subset.
    system, parameter_value = system_subset(
        parameter_key=ParameterGradientKey("vdW", "[#1:1]", "epsilon"),
        force_field=hydrogen_chloride_force_field(True, True, False),
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    charge_0, sigma_0, epsilon_0 = system.getForce(0).getParticleParameters(0)
    charge_1, sigma_1, epsilon_1 = system.getForce(0).getParticleParameters(1)

    assert np.isclose(charge_0.value_in_unit(openmm_unit.elementary_charge), 0.0)
    assert np.isclose(charge_1.value_in_unit(openmm_unit.elementary_charge), 0.0)

    assert np.isclose(sigma_0.value_in_unit(openmm_unit.angstrom), 2.0)
    assert np.isclose(sigma_1.value_in_unit(openmm_unit.angstrom), 1.0)

    assert np.isclose(epsilon_0.value_in_unit(openmm_unit.kilojoules_per_mole), 2.0)
    assert np.isclose(epsilon_1.value_in_unit(openmm_unit.kilojoules_per_mole), 0.5)


def test_system_subset_vdw_cutoff():
    """Test that handler attributes are correctly handled."""

    # Create a dummy topology
    topology: Topology = Molecule.from_smiles("Cl").to_topology()
    topology.box_vectors = numpy.eye(3) * openmm_unit.nanometers

    # Create the system subset.
    system, parameter_value = system_subset(
        parameter_key=ParameterGradientKey("vdW", None, "cutoff"),
        force_field=hydrogen_chloride_force_field(True, True, False),
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    cutoff = system.getForce(0).getCutoffDistance()
    assert np.isclose(cutoff.value_in_unit(openmm_unit.angstrom), 9.0)


def test_system_subset_library_charge():

    force_field = hydrogen_chloride_force_field(True, False, False)

    # Ensure a zero charge after perturbation.
    force_field.get_parameter_handler("LibraryCharges").parameters["[#1:1]"].charge1 = (
        1.5 * openmm_unit.elementary_charge
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

    assert np.isclose(charge_0.value_in_unit(openmm_unit.elementary_charge), -1.5)
    assert np.isclose(charge_1.value_in_unit(openmm_unit.elementary_charge), 1.5)

    assert np.isclose(sigma_0.value_in_unit(openmm_unit.angstrom), 10.0)
    assert np.isclose(sigma_1.value_in_unit(openmm_unit.angstrom), 10.0)

    assert np.isclose(epsilon_0.value_in_unit(openmm_unit.kilojoules_per_mole), 0.0)
    assert np.isclose(epsilon_1.value_in_unit(openmm_unit.kilojoules_per_mole), 0.0)


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
        force_field=hydrogen_chloride_force_field(False, True, False),
        topology=topology,
        scale_amount=0.5,
    )

    assert system.getNumForces() == 1
    assert system.getNumParticles() == 2

    charge_0, sigma_0, epsilon_0 = system.getForce(0).getParticleParameters(0)
    charge_1, sigma_1, epsilon_1 = system.getForce(0).getParticleParameters(1)

    assert not np.isclose(charge_0.value_in_unit(openmm_unit.elementary_charge), -1.0)
    assert np.isclose(charge_1.value_in_unit(openmm_unit.elementary_charge), 1.0)

    assert np.isclose(sigma_0.value_in_unit(openmm_unit.angstrom), 10.0)
    assert np.isclose(sigma_1.value_in_unit(openmm_unit.angstrom), 10.0)

    assert np.isclose(epsilon_0.value_in_unit(openmm_unit.kilojoules_per_mole), 0.0)
    assert np.isclose(epsilon_1.value_in_unit(openmm_unit.kilojoules_per_mole), 0.0)


@pytest.mark.parametrize(
    "smirks, all_zeros", [("[#6X4:1]", True), ("[#8:1]", False), (None, False)]
)
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
        [
            ParameterGradientKey(
                "vdW", smirks, "epsilon" if smirks is not None else "cutoff"
            )
        ],
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
            0.0 * observables["PotentialEnergy"].gradients[0].value.units,
        )
    else:
        assert not np.allclose(
            observables["PotentialEnergy"].gradients[0].value,
            0.0 * observables["PotentialEnergy"].gradients[0].value.units,
        )


@pytest.mark.parametrize(
    "box_vectors", [None, (numpy.eye(3) * 3.0) * openmm_unit.nanometers]
)
def test_update_context_with_positions(box_vectors):

    force_field = hydrogen_chloride_force_field(True, False, True)

    topology: Topology = Molecule.from_smiles("Cl").to_topology()
    system = force_field.create_openmm_system(topology)

    context = openmm.Context(
        system, openmm.VerletIntegrator(0.1 * openmm_unit.femtoseconds)
    )

    positions = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * openmm_unit.angstrom

    update_context_with_positions(context, positions, box_vectors)

    context_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    context_box_vectors = context.getState(getPositions=True).getPeriodicBoxVectors()

    assert numpy.allclose(
        context_positions.value_in_unit(openmm_unit.angstrom),
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    assert numpy.isclose(
        context_box_vectors[0].x, (2.0 if box_vectors is None else 3.0)
    )
    assert numpy.isclose(
        context_box_vectors[1].y, (2.0 if box_vectors is None else 3.0)
    )
    assert numpy.isclose(
        context_box_vectors[2].z, (2.0 if box_vectors is None else 3.0)
    )


def test_update_context_with_pdb(tmpdir):

    force_field = hydrogen_chloride_force_field(True, False, True)

    topology: Topology = Molecule.from_smiles("Cl").to_topology()
    system = force_field.create_openmm_system(topology)

    context = openmm.Context(
        system, openmm.VerletIntegrator(0.1 * openmm_unit.femtoseconds)
    )

    positions = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * openmm_unit.angstrom

    pdb_path = os.path.join(tmpdir, "tmp.pdb")
    topology.to_file(pdb_path, positions)

    pdb_file = PDBFile(pdb_path)

    update_context_with_pdb(context, pdb_file)

    context_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    context_box_vectors = context.getState(getPositions=True).getPeriodicBoxVectors()

    assert numpy.allclose(
        context_positions.value_in_unit(openmm_unit.angstrom),
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    assert numpy.allclose(
        extract_positions(context.getState(getPositions=True), [2]).value_in_unit(
            openmm_unit.angstrom
        ),
        numpy.array([[2.0, 0.0, 0.0]]),
    )

    assert numpy.isclose(context_box_vectors[0].x, 2.0)
    assert numpy.isclose(context_box_vectors[1].y, 2.0)
    assert numpy.isclose(context_box_vectors[2].z, 2.0)


def test_extract_atom_indices():

    force_field = hydrogen_chloride_force_field(True, False, True)

    topology: Topology = Molecule.from_smiles("Cl").to_topology()
    system = force_field.create_openmm_system(topology)

    assert system.getNumParticles() == 3
    assert extract_atom_indices(system) == [0, 1]
