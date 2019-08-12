import pkg_resources
import yaml

from propertyestimator.properties import Density
from propertyestimator.protocols import coordinates, groups, simulation
from propertyestimator.substances import Substance


def create_debug_density_workflow(max_molecules=128,
                                  equilibration_steps=50,
                                  equilibration_frequency=5,
                                  production_steps=100,
                                  production_frequency=5):

    density_workflow_schema = Density.get_default_simulation_workflow_schema()

    build_coordinates = coordinates.BuildCoordinatesPackmol('')
    build_coordinates.schema = density_workflow_schema.protocols['build_coordinates']

    build_coordinates.max_molecules = max_molecules

    density_workflow_schema.protocols['build_coordinates'] = build_coordinates.schema

    npt_equilibration = simulation.RunOpenMMSimulation('')
    npt_equilibration.schema = density_workflow_schema.protocols['npt_equilibration']

    npt_equilibration.steps = equilibration_steps
    npt_equilibration.output_frequency = equilibration_frequency

    density_workflow_schema.protocols['npt_equilibration'] = npt_equilibration.schema

    converge_uncertainty = groups.ConditionalGroup('')
    converge_uncertainty.schema = density_workflow_schema.protocols['converge_uncertainty']

    converge_uncertainty.protocols['npt_production'].steps = production_steps
    converge_uncertainty.protocols['npt_production'].output_frequency = production_frequency

    density_workflow_schema.protocols['converge_uncertainty'] = converge_uncertainty.schema

    return density_workflow_schema


def mol2_to_smiles(file_path):
    """Loads a receptor from a mol2 file.

    Parameters
    ----------
    file_path: str
        The file path to the mol2 file.

    Returns
    -------
    str
        The smiles descriptor of the loaded receptor molecule
    """
    from openforcefield.topology import Molecule

    receptor_molecule = Molecule.from_file(file_path, 'MOL2')
    return receptor_molecule.to_smiles()


def build_substance(ligand_smiles, receptor_smiles, ionic_strength=None):
    """Builds a substance containing a ligand and receptor solvated
    in an aqueous solution with a given ionic strength

    Parameters
    ----------
    ligand_smiles: str, optional
        The smiles descriptor of the ligand.
    receptor_smiles: str
        The smiles descriptor of the host.
    ionic_strength: simtk.unit.Quantity, optional
        The ionic strength of the aqueous solvent.

    Returns
    -------
    Substance
        The built substance.
    """

    substance = Substance()

    if ligand_smiles is not None:

        ligand = Substance.Component(smiles=ligand_smiles,
                                     role=Substance.ComponentRole.Ligand)

        substance.add_component(component=ligand, amount=Substance.ExactAmount(1))

    receptor = Substance.Component(smiles=receptor_smiles,
                                   role=Substance.ComponentRole.Receptor)

    substance.add_component(component=receptor, amount=Substance.ExactAmount(1))

    water = Substance.Component(smiles='O', role=Substance.ComponentRole.Solvent)
    sodium = Substance.Component(smiles='[Na+]', role=Substance.ComponentRole.Solvent)
    chlorine = Substance.Component(smiles='[Cl-]', role=Substance.ComponentRole.Solvent)

    water_mole_fraction = 1.0

    if ionic_strength is not None:

        salt_mole_fraction = Substance.calculate_aqueous_ionic_mole_fraction(ionic_strength) / 2.0
        water_mole_fraction = 1.0 - salt_mole_fraction

        substance.add_component(component=sodium, amount=Substance.MoleFraction(salt_mole_fraction))
        substance.add_component(component=chlorine, amount=Substance.MoleFraction(salt_mole_fraction))

    substance.add_component(component=water, amount=Substance.MoleFraction(water_mole_fraction))

    return substance


def get_paprika_host_guest_substance(host_name, guest_name, ionic_strength=None):

    installed_benchmarks = {}

    for entry_point in pkg_resources.iter_entry_points(group="taproom.benchmarks"):
        installed_benchmarks[entry_point.name] = entry_point.load()


    substances = []
    orientations = []
    host_yaml_paths = []

    for orientation, yaml_path in installed_benchmarks["host_guest_systems"][host_name]["yaml"].items():
            host_yaml_paths.append(yaml_path)
            orientations.append(orientation)

    for host_yaml_path in host_yaml_paths:

        with open(host_yaml_path, "r") as file:
            host_yaml = yaml.safe_load(file)

        host_mol2_path = str(host_yaml_path.parent.joinpath(
            host_yaml['structure']))

        host_smiles = mol2_to_smiles(host_mol2_path)
        guest_smiles = None

        if guest_name is not None:

            guest_yaml_path = installed_benchmarks["host_guest_systems"][host_name][guest_name]["yaml"]

            with open(guest_yaml_path, "r") as file:
                guest_yaml = yaml.safe_load(file)

            guest_mol2_path = str(host_yaml_path.parent.joinpath(
                                  guest_name).joinpath(
                                  guest_yaml['structure']))

            guest_smiles = mol2_to_smiles(guest_mol2_path)

        substances.append(build_substance(guest_smiles, host_smiles, ionic_strength))

    return substances, orientations
