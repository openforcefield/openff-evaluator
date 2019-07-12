import os
import shutil
from enum import Enum

from propertyestimator import server
from propertyestimator.backends import DaskLocalClusterBackend, ComputeResources, QueueWorkerResources, DaskLSFBackend
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import Density
from propertyestimator.properties import PropertyPhase, CalculationSource, DielectricConstant, EnthalpyOfMixing
from propertyestimator.protocols import coordinates, groups, simulation
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.workflow import WorkflowOptions
from simtk import unit


class BackendType(Enum):
    LocalCPU = 'LocalCPU'
    LocalGPU = 'LocalGPU'
    GPU = 'GPU'
    CPU = 'CPU'


def setup_server(backend_type=BackendType.LocalCPU, max_number_of_workers=1, conda_environment='propertyestimator'):

    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    # Remove any existing data.
    if os.path.isdir(working_directory):
        shutil.rmtree(working_directory)

    calculation_backend = None

    if backend_type == BackendType.LocalCPU:
        calculation_backend = DaskLocalClusterBackend(number_of_workers=1)

    elif backend_type == BackendType.LocalGPU:

        calculation_backend = DaskLocalClusterBackend(number_of_workers=1,
                                                      resources_per_worker=ComputeResources(1,
                                                                                            1,
                                                                                            ComputeResources.
                                                                                            GPUToolkit.CUDA))

    elif backend_type == BackendType.GPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               number_of_gpus=1,
                                               preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                               per_thread_memory_limit=8 * (unit.giga * unit.byte),
                                               wallclock_time_limit="05:59")

        worker_script_commands = [
            f'export OE_LICENSE="/home/boothros/oe_license.txt"',
            f'. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            f'conda activate {conda_environment}',
            f'module load cuda/9.2'
        ]

        extra_script_options = [
            '-m "ls-gpu lt-gpu"'
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='gpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             extra_script_options=extra_script_options,
                                             adaptive_interval='1000ms')
    elif backend_type == BackendType.CPU:

        queue_resources = QueueWorkerResources(number_of_threads=1,
                                               per_thread_memory_limit=10 * (unit.giga * unit.byte),
                                               wallclock_time_limit="01:30")

        worker_script_commands = [
            f'export OE_LICENSE="/home/boothros/oe_license.txt"',
            f'. /home/boothros/miniconda3/etc/profile.d/conda.sh',
            f'conda activate {conda_environment}',
        ]

        calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                             maximum_number_of_workers=max_number_of_workers,
                                             resources_per_worker=queue_resources,
                                             queue_name='cpuqueue',
                                             setup_script_commands=worker_script_commands,
                                             adaptive_interval='1000ms')

    storage_backend = LocalFileStorage(storage_directory)

    server.PropertyEstimatorServer(calculation_backend=calculation_backend,
                                   storage_backend=storage_backend,
                                   working_directory=working_directory)


def create_debug_density_workflow(max_molecules=128,
                                  equilibration_steps=50,
                                  equilibration_frequency=5,
                                  production_steps=100,
                                  production_frequency=5):

    density_workflow_schema = Density.get_default_simulation_workflow_schema(WorkflowOptions())

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
        elements = ['C']

    substance = Substance()

    mole_fraction = 1.0 / number_of_components

    for index in range(number_of_components):

        smiles_pattern = ''.join(elements * (index + 1))

        substance.add_component(Substance.Component(smiles_pattern),
                                Substance.MoleFraction(mole_fraction))

    return substance


def create_dummy_property(property_class):
    """Create a dummy liquid property of specified type measured at
    298 K and 1 atm, with 2 components of methane and ethane.

    The property also contains a dummy receptor metadata entry.

    Parameters
    ----------
    property_class : type
        The type of property, e.g. Density, DielectricConstant...

    Returns
    -------
    PhysicalProperty
        The created property.
    """
    substance = create_dummy_substance(number_of_components=2)

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram)
    
    dummy_property.source = CalculationSource(fidelity='dummy', provenance={})

    # Make sure the property has the meta data required for more
    # involved properties.
    dummy_property.metadata = {
        'receptor_mol2': 'unknown_path.mol2'
    }

    return dummy_property


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

    source = CalculationSource('Dummy', {})
    carbon_substance = create_dummy_substance(number_of_components=1, elements=['C'])

    density_property = Density(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                      pressure=0.5 * unit.atmosphere),
                               phase=PropertyPhase.Liquid,
                               substance=carbon_substance,
                               value=1 * unit.gram / unit.milliliter,
                               uncertainty=0.11 * unit.gram / unit.milliliter,
                               source=source)

    nitrogen_substance = create_dummy_substance(number_of_components=2, elements=['N'])

    dielectric_property = DielectricConstant(thermodynamic_state=ThermodynamicState(temperature=288 * unit.kelvin,
                                                                                    pressure=1 * unit.atmosphere),
                                             phase=PropertyPhase.Gas,
                                             substance=nitrogen_substance,
                                             value=1 * unit.dimensionless,
                                             uncertainty=0.11 * unit.dimensionless,
                                             source=source)

    oxygen_substance = create_dummy_substance(number_of_components=3, elements=['O'])

    enthalpy_property = EnthalpyOfMixing(thermodynamic_state=ThermodynamicState(temperature=308 * unit.kelvin,
                                                                                pressure=1.5 * unit.atmosphere),
                                         phase=PropertyPhase.Solid,
                                         substance=oxygen_substance,
                                         value=1 * unit.kilojoules_per_mole,
                                         uncertainty=0.11 * unit.kilojoules_per_mole,
                                         source=source)

    data_set = PhysicalPropertyDataSet()
    data_set.properties[carbon_substance.identifier] = [density_property]
    data_set.properties[nitrogen_substance.identifier] = [dielectric_property]
    data_set.properties[oxygen_substance.identifier] = [enthalpy_property]

    return data_set


def build_tip3p_smirnoff_force_field(file_path=None):
    """Combines the smirnoff99Frosst and tip3p offxml files
    into a single one which can be consumed by the property
    estimator.

    Parameters
    ----------
    file_path: str, optional
        The path to save the force field to.

    Returns
    -------
    str
        The file path to the combined force field file.
    """
    from openforcefield.typing.engines.smirnoff import ForceField
    
    smirnoff_force_field_path = get_data_filename('forcefield/smirnoff99Frosst.offxml')
    tip3p_force_field_path = get_data_filename('forcefield/tip3p.offxml')

    smirnoff_force_field_with_tip3p = ForceField(smirnoff_force_field_path,
                                                 tip3p_force_field_path)

    force_field_path = 'smirnoff99Frosst_tip3p.offxml' if file_path is None else file_path
    smirnoff_force_field_with_tip3p.to_file(force_field_path)

    return force_field_path
