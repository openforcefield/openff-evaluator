"""
A collection of osmotic physical property definitions.
"""

import copy
import logging
import pathlib
import tempfile

import mdtraj
import numpy
import openmm
import openmm.app
import openmm.unit
import scipy.spatial.distance
from openff.toolkit import Molecule
from openff.units import unit

from openff.evaluator.backends import ComputeResources
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.forcefield import ForceFieldSource, SmirnoffForceFieldSource
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.protocols.openmm import OpenMMSimulation
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.thermodynamics import Ensemble, ThermodynamicState
from openff.evaluator.utils import packmol
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath

_LOGGER = logging.getLogger(__name__)

NM = unit.nanometers

M = unit.molar
K = unit.kelvin


def _parameterize(
    ff_source: ForceFieldSource,
    coords: mdtraj.Trajectory,
    substance: Substance,
    output_dir: pathlib.Path,
    resources: ComputeResources,
) -> ParameterizedSystem:
    output_dir.mkdir(exist_ok=True, parents=True)

    if not isinstance(ff_source, SmirnoffForceFieldSource):
        raise NotImplementedError("only SMIRNOFF force fields are supported.")

    ff_source.json(str(output_dir / "ff.json"))
    coords.save(str(output_dir / "coords.pdb"))

    protocol = BuildSmirnoffSystem("parameterize")
    protocol.force_field_path = "ff.json"
    protocol.coordinate_file_path = "coords.pdb"
    protocol.substance = substance
    protocol.execute(str(output_dir), resources)

    return protocol.parameterized_system


def _simulate(
    system: ParameterizedSystem,
    coords: mdtraj.Trajectory,
    temperature: unit.Quantity,
    pressure: unit.Quantity | None,
    n_steps: int,
    timestep: unit.Quantity,
    friction: unit.Quantity,
    output_freq: int,
    checkpoint_freq: int,
    output_dir: pathlib.Path,
    resources: ComputeResources,
) -> mdtraj.Trajectory:
    output_dir.mkdir(exist_ok=True, parents=True)
    coords[-1].save(str(output_dir / "input-coords.pdb"))

    protocol = OpenMMSimulation("simulation")
    protocol.steps_per_iteration = n_steps
    protocol.total_number_of_iterations = 1
    protocol.output_frequency = output_freq
    protocol.checkpoint_frequency = checkpoint_freq
    protocol.timestep = timestep
    protocol.thermodynamic_state = ThermodynamicState(
        temperature=temperature, pressure=pressure
    )
    protocol.ensemble = Ensemble.NPT if pressure is not None else Ensemble.NVT
    protocol.thermostat_friction = friction
    protocol.input_coordinate_file = "input-coords.pdb"
    protocol.parameterized_system = system
    protocol.enable_pbc = True

    protocol.execute(str(output_dir), resources)

    return mdtraj.load(protocol.output_coordinate_file)


def _extract_vdw_radii(topology: openmm.app.Topology) -> numpy.ndarray:
    return (
        numpy.array(
            [
                oechem.OEGetBondiVdWRadius(atom.element.atomic_number)
                for atom in topology.atoms()
            ]
        )
        / 10.0
    )  # ang to nm


def _remove_clashes(
    topology_ions: openmm.app.Topology,
    coords_ions: unit.Quantity,
    topology_pure: openmm.app.Topology,
    coords_pure: unit.Quantity,
    box_offset: numpy.ndarray,
) -> tuple[openmm.app.Topology, unit.Quantity]:
    coords_ions = coords_ions.value_in_unit(NM)
    coords_ions -= coords_ions.mean(axis=0)
    coords_pure = coords_pure.value_in_unit(NM)
    coords_pure -= coords_pure.mean(axis=0)

    radii_ions = _extract_vdw_radii(topology_ions)
    radii_pure = _extract_vdw_radii(topology_pure)

    diam = radii_pure[:, None] + radii_ions[None, :]

    distances_minus = scipy.spatial.distance.cdist(
        coords_pure - box_offset, coords_ions
    )
    clashes_minus = (distances_minus < diam).any(axis=1)

    distances_plus = scipy.spatial.distance.cdist(coords_pure + box_offset, coords_ions)
    clashes_plus = (distances_plus < diam).any(axis=1)

    clashes = numpy.repeat((clashes_minus | clashes_plus).reshape(-1, 3).all(axis=1), 3)

    n_removed = clashes.sum() // 3

    _LOGGER.info(f"removing {n_removed} water molecules due to clashes.")

    coords_pure = coords_pure[~clashes]

    coords = numpy.vstack([coords_pure + box_offset, coords_ions])
    n_water = len(coords_pure) // 3

    topology = openmm.app.Topology()
    chain = topology.addChain()

    residue_pure = next(iter(topology_pure.residues()))

    for _ in range(n_water):
        residue = topology.addResidue(residue_pure.name, chain)

        for atom in residue_pure.atoms():
            topology.addAtom(atom.name, atom.element, residue)

    for chain_orig in topology_ions.chains():
        for residue_orig in chain_orig.residues():
            residue = topology.addResidue(residue_orig.name, chain)

            for atom in residue_orig.atoms():
                topology.addAtom(atom.name, atom.element, residue)

    with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        openmm.app.PDBFile.writeFile(topology, coords * NM, temp.name)
        topology = openmm.app.PDBFile(temp.name).topology

    return topology, coords * NM


def setup_system(
    concentration_molar: float,
    solvent: Molecule,
    anion: Molecule,
    cation: Molecule,
    ff_source: ForceFieldSource,
    output_dir: pathlib.Path,
    resources: ComputeResources,
):
    output_dir.mkdir(exist_ok=True, parents=True)

    cutoff = 1.2 * NM
    switch = 1.0 * NM

    temperature = 300.0 * K
    concentration = concentration_molar * M

    box_length = 4.8 * NM

    box_size = numpy.array([box_length.value_in_unit(NM)] * 3) * NM
    box_offset = numpy.array([0.0, 0.0, box_length.value_in_unit(NM)])

    n_solvent = ...
    n_ion_pairs = ...

    _LOGGER.info(f"creating initial box with ions.")

    trajectory_0, assigned_residue_names = packmol.pack_box(
        molecules=[solvent, anion, cation],
        number_of_copies=[n_solvent, n_ion_pairs, n_ion_pairs],
        structure_to_solvate=None,
        box_size=box_size,
    )
    topology: mdtraj.Topology = trajectory_0.topology

    substance = Substance()
    substance.add_component(Component(smiles=solvent.to_smiles()), MoleFraction(1.0))
    substance.add_component(
        Component(smiles=anion.to_smiles()), ExactAmount(n_ion_pairs)
    )
    substance.add_component(
        Component(smiles=cation.to_smiles()), ExactAmount(n_ion_pairs)
    )

    _LOGGER.info(f"initial box contains {topology.n_atoms} atoms.")
    _LOGGER.info(f"parameterizing system.")

    system = _parameterize(
        ff_source, trajectory_0, substance, output_dir / "parameterize-ions", resources
    )

    topology_ions = system.topology.to_openmm()

    _LOGGER.info(f"equilibrating ion box.")

    trajectory_ions = _simulate(
        system,
        trajectory_0,
        temperature,
        pressure=None,
        n_steps=200000,
        timestep=1.0 * unit.femtosecond,
        friction=1.0 / unit.picosecond,
        output_freq=20000,
        checkpoint_freq=1,
        output_dir=output_dir / "equilibrate",
        resources=resources,
    )
    coords_ions = trajectory_ions.xyz[0] * NM

    _LOGGER.info(f"creating pure water box.")

    modeller = openmm.app.Modeller(openmm.app.Topology(), [])
    modeller.addSolvent(
        openmm.app.ForceField("tip3p.xml"), model="tip3p", boxSize=box_size
    )
    _LOGGER.info(
        f"pure water box contains {modeller.topology.getNumAtoms() // 3} waters."
    )

    topology_pure = modeller.topology
    coords_pure = (
        numpy.array(modeller.positions.value_in_unit(openmm.unit.nanometer)) * NM
    )

    topology, coords = _remove_clashes(
        topology_ions, coords_ions, topology_pure, coords_pure, box_offset
    )
    topology.setUnitCellDimensions(
        [
            box_length.m_as(NM),
            box_length.m_as(NM),
            box_length.m_as(NM) * 2.0,
        ]
        * openmm.unit.nanometer
    )

    trajectory = mdtraj.Trajectory(
        coords.m_as(NM), mdtraj.Topology.from_openmm(topology)
    )

    param_system = _parameterize(
        ff_source,
        trajectory,
        substance,
        output_dir / "parameterize-full",
        resources,
    )

    param_system_frozen = copy.deepcopy(param_system)
    system_frozen = param_system_frozen.system

    counter = 0
    ion_idxs = []

    for chain in topology.chains():
        for residue in chain.residues():
            for _ in residue.atoms():
                if residue.name != "HOH":
                    system_frozen.setParticleMass(counter, 0.0)
                    ion_idxs.append(counter)

                counter += 1

    (output_dir / "parameterize-full" / "system-frozen.xml").write_text(
        openmm.XmlSerializer.serialize(system_frozen)
    )
    param_system_frozen._system_path = system_frozen

    _LOGGER.info(f"equilibrating with {len(ion_idxs)} frozen ions.")

    trajectory = _simulate(
        param_system_frozen,
        trajectory,
        temperature,
        None,
        200000,
        1.0 * unit.femtosecond,
        1.0 / unit.picosecond,
        20000,
        1,
        output_dir / "equilibrate-full",
        resources,
    )

    _LOGGER.info(f"adding z-restraint.")

    restraint_group = 16

    force_groups = {force.getForceGroup() for force in system.getForces()}
    assert restraint_group not in force_groups

    restraint_force = create_flat_bottom_restraint(ion_idxs)
    restraint_force.setForceGroup(restraint_group)
    system.addForce(restraint_force)

    barostat = openmm.MonteCarloAnisotropicBarostat(
        openmm.Vec3(1.0, 1.0, 1.0) * openmm.unit.atmosphere,
        temperature,
        False,
        False,
        True,
        25,
    )

    system_npt = copy.deepcopy(system)
    system_npt.addForce(barostat)

    _LOGGER.info(f"running npt equilibration.")

    coords, box = _equilibrate(
        topology,
        system_npt,
        coords,
        box,
        time=500.0 * openmm.unit.picosecond,
        temperature=temperature,
        reporters=[
            openmm.app.DCDReporter(
                str(output_dir / "equil.dcd"), 1000, enforcePeriodicBox=True
            )
        ],
    )
    topology.setPeriodicBoxVectors(box)

    openmm.app.PDBFile.writeFile(topology, coords, str(output_dir / "equil.pdb"))

    timestep = 2.0 * openmm.unit.femtosecond
    time_prod = 20.0 * openmm.unit.nanosecond

    simulation = _create_simulation(
        topology, system, coords, box, timestep, temperature=temperature
    )

    interval = 1.0 * openmm.unit.picosecond
    n_steps_per_interval = int(
        interval.value_in_unit(openmm.unit.nanosecond)
        / timestep.value_in_unit(openmm.unit.nanosecond)
    )

    simulation.reporters.append(
        openmm.app.DCDReporter(
            str(output_dir / "prod.dcd"), n_steps_per_interval, enforcePeriodicBox=True
        )
    )

    n_intervals = int(
        time_prod.value_in_unit(openmm.unit.nanosecond)
        / interval.value_in_unit(openmm.unit.nanosecond)
    )

    _LOGGER.info(f"running production for {n_intervals * n_steps_per_interval} steps.")

    forces_restraint = []

    for _ in tqdm.tqdm(range(n_intervals), total=n_intervals):
        simulation.step(n_steps_per_interval)

        state_restraint: openmm.State = simulation.context.getState(
            getForces=True, groups={restraint_group}
        )
        forces_restraint.append(
            numpy.abs(
                state_restraint.getForces(asNumpy=True).value_in_unit(KCAL / ANG)
            ).sum(axis=0)
        )

    forces_restraint = numpy.vstack(forces_restraint)
    numpy.save(str(output_dir / "forces.npy"), forces_restraint, allow_pickle=False)


@thermoml_property("Osmotic coefficient", supported_phases=PropertyPhase.Liquid)
class OsmoticCoefficient(PhysicalProperty):
    def default_unit(cls):
        return unit.dimensionless

    @staticmethod
    def default_simulation_schema():
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = SimulationSchema()

        ...

        # Build the workflow schema.
        schema = WorkflowSchema()
        schema.protocol_schemas = [...]
        schema.outputs_to_store = {"full_system": output_to_store}
        schema.final_value_source = value_source

        calculation_schema.workflow_schema = schema
        return calculation_schema


# Register the properties via the plugin system.
register_calculation_schema(
    OsmoticCoefficient, SimulationLayer, OsmoticCoefficient.default_simulation_schema
)
