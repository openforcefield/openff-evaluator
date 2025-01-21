import os
import pathlib
import shutil
import uuid
from contextlib import contextmanager

from openff.toolkit import ForceField
from openff.units import unit
from openff.utilities.utilities import get_data_dir_path

from openff.evaluator.datasets import (
    CalculationSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import Density, DielectricConstant, EnthalpyOfMixing
from openff.evaluator.storage.data import StoredSimulationData
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import ObservableFrame


@contextmanager
def does_not_raise():
    """A helpful context manager to use inplace of a pytest raise statement
    when no exception is expected."""
    yield


def create_dummy_substance(number_of_components, elements=None):
    """Creates a substance with a given number of components,
    each containing the specified elements.

    Parameters
    ----------
    number_of_components : int
        The number of components to add to the substance.
    elements : list of str
        The elements that each component should containt.

    Returns
    -------
    Substance
        The created substance.
    """
    if elements is None:
        elements = ["C"]

    substance = Substance()

    mole_fraction = 1.0 / number_of_components

    for index in range(number_of_components):
        smiles_pattern = "".join(elements * (index + 1))

        substance.add_component(Component(smiles_pattern), MoleFraction(mole_fraction))

    return substance


def create_dummy_property(property_class):
    """Create a dummy liquid property of specified type measured at
    298 K and 1 atm, with 2 components of methane and ethane.

    The property also contains a dummy receptor metadata entry.

    Parameters
    ----------
    property_class : type of PhysicalProperty
        The type of property, e.g. Density, DielectricConstant...

    Returns
    -------
    PhysicalProperty
        The created property.
    """
    substance = create_dummy_substance(number_of_components=2)

    dummy_property = property_class(
        thermodynamic_state=ThermodynamicState(
            temperature=298 * unit.kelvin, pressure=1 * unit.atmosphere
        ),
        phase=PropertyPhase.Liquid,
        substance=substance,
        value=10.0 * property_class.default_unit(),
        uncertainty=1.0 * property_class.default_unit(),
    )

    dummy_property.source = CalculationSource(fidelity="dummy", provenance={})

    # Make sure the property has the meta data required for more
    # involved properties.
    dummy_property.metadata = {"receptor_mol2": "unknown_path.mol2"}

    return dummy_property


def create_dummy_simulation_data(
    directory_path,
    substance,
    force_field_id="dummy_ff_id",
    coordinate_file_name="output.pdb",
    trajectory_file_name="trajectory.dcd",
    statistical_inefficiency=1.0,
    phase=PropertyPhase.Liquid,
    number_of_molecules=1,
    calculation_id=None,
):
    """Creates a dummy `StoredSimulationData` object and
    the corresponding data directory.

    Parameters
    ----------
    directory_path: str
        The path to the dummy data directory to create.
    substance: Substance
    force_field_id
    coordinate_file_name
    trajectory_file_name
    statistics_file_name
    statistical_inefficiency
    phase
    number_of_molecules
    calculation_id

    Returns
    -------
    StoredSimulationData
        The dummy stored data object.
    """

    os.makedirs(directory_path, exist_ok=True)

    data = StoredSimulationData()

    data.substance = substance
    data.force_field_id = force_field_id
    data.thermodynamic_state = ThermodynamicState(1.0 * unit.kelvin)
    data.property_phase = phase

    data.coordinate_file_name = coordinate_file_name
    data.trajectory_file_name = trajectory_file_name
    data.observables = ObservableFrame()

    with open(os.path.join(directory_path, coordinate_file_name), "w") as file:
        file.write("")
    with open(os.path.join(directory_path, trajectory_file_name), "w") as file:
        file.write("")

    data.statistical_inefficiency = statistical_inefficiency

    data.number_of_molecules = number_of_molecules

    if calculation_id is None:
        calculation_id = str(uuid.uuid4())

    data.source_calculation_id = calculation_id

    return data


def create_filterable_data_set():
    """Creates a dummy data with a diverse set of properties to
    be filtered, namely:

        - a liquid density measured at 298 K and 0.5 atm with 1 component containing only carbon.
        - a gaseous dielectric measured at 288 K and 1 atm with 2 components containing only nitrogen.
        - a solid EoM measured at 308 K and 1.5 atm with 3 components containing only oxygen.

    Returns
    -------
    PhysicalPropertyDataSet
        The created data set.
    """

    source = CalculationSource("Dummy", {})
    carbon_substance = create_dummy_substance(number_of_components=1, elements=["C"])

    density_property = Density(
        thermodynamic_state=ThermodynamicState(
            temperature=298 * unit.kelvin, pressure=0.5 * unit.atmosphere
        ),
        phase=PropertyPhase.Liquid,
        substance=carbon_substance,
        value=1 * unit.gram / unit.milliliter,
        uncertainty=0.11 * unit.gram / unit.milliliter,
        source=source,
    )

    nitrogen_substance = create_dummy_substance(number_of_components=2, elements=["N"])

    dielectric_property = DielectricConstant(
        thermodynamic_state=ThermodynamicState(
            temperature=288 * unit.kelvin, pressure=1 * unit.atmosphere
        ),
        phase=PropertyPhase.Gas,
        substance=nitrogen_substance,
        value=1 * unit.dimensionless,
        uncertainty=0.11 * unit.dimensionless,
        source=source,
    )

    oxygen_substance = create_dummy_substance(number_of_components=3, elements=["O"])

    enthalpy_property = EnthalpyOfMixing(
        thermodynamic_state=ThermodynamicState(
            temperature=308 * unit.kelvin, pressure=1.5 * unit.atmosphere
        ),
        phase=PropertyPhase.Solid,
        substance=oxygen_substance,
        value=1 * unit.kilojoules / unit.mole,
        uncertainty=0.11 * unit.kilojoules / unit.mole,
        source=source,
    )

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(density_property, dielectric_property, enthalpy_property)

    return data_set


def build_tip3p_smirnoff_force_field():
    """Combines the smirnoff99Frosst and tip3p offxml files
    into a single one which can be consumed by the property
    estimator.

    Returns
    -------
    SmirnoffForceFieldSource
        The force field containing both smirnoff99Frosst-1.1.0
        and TIP3P parameters
    """
    from openff.toolkit.typing.engines.smirnoff import ForceField

    smirnoff_force_field_path = "smirnoff99Frosst-1.1.0.offxml"
    tip3p_force_field_path = get_data_filename("forcefield/tip3p.offxml")

    smirnoff_force_field_with_tip3p = ForceField(
        smirnoff_force_field_path, tip3p_force_field_path
    )

    return SmirnoffForceFieldSource.from_object(smirnoff_force_field_with_tip3p)


def _write_force_field(force_field: str = "openff-2.0.0.offxml"):
    """
    Write a force field file to disk.
    """
    ff = ForceField(force_field)
    with open("force-field.json", "w") as file:
        file.write(SmirnoffForceFieldSource.from_object(ff).json())


def _copy_property_working_data(
    source_directory: str,
    uuid_prefix: str | list[str],
    destination_directory: str = ".",
    include_data_files: bool = False,
):
    """
    Copy test data from one directory to another.
    This is typically working data in a *Layer directory.

    Parameters
    ----------
    source_directory : str
        The directory to copy data from.
        This should be relative to to the data directory.
        For tests, the path will typically start with `test`.
    uuid_prefix : str or list[str]
        The prefix of the UUID/s to copy.
    destination_directory : str
        The directory to copy data to.
        This should be relative to the current working directory.
        Default is the current working directory.
    """
    if isinstance(uuid_prefix, str):
        uuid_prefix = [uuid_prefix]

    # locate our saved test data
    data_directory = pathlib.Path(
        get_data_dir_path(source_directory, "openff.evaluator")
    )
    abs_path = data_directory.resolve()
    destination_directory = pathlib.Path(destination_directory)

    # copy data files over from data_directory
    for path in abs_path.iterdir():
        if path.name.startswith(tuple(uuid_prefix)) and path.is_dir():
            dest_dir = destination_directory / path.name
            shutil.copytree(path, dest_dir)

    if include_data_files:
        for path in abs_path.glob("data*.json"):
            shutil.copy(path, destination_directory)
