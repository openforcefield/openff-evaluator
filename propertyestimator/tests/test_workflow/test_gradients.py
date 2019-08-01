"""
Units tests for propertyestimator.workflow
"""
import tempfile
from os import path

from openforcefield.typing.engines import smirnoff
from simtk import unit

from propertyestimator.backends import ComputeResources, DaskLocalCluster
from propertyestimator.client import PropertyEstimatorClient, PropertyEstimatorOptions, ConnectionOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import CalculationSource, PropertyPhase, ParameterGradientKey, \
    Density
from propertyestimator.protocols import coordinates, simulation, groups
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow import WorkflowOptions


def create_debug_density_workflow(max_molecules=128,
                                  mass_density=0.95 * unit.grams / unit.milliliters,
                                  equilibration_steps=50,
                                  equilibration_frequency=5,
                                  production_steps=100,
                                  production_frequency=5):

    density_workflow_schema = Density.get_default_simulation_workflow_schema(WorkflowOptions())

    build_coordinates = coordinates.BuildCoordinatesPackmol('')
    build_coordinates.schema = density_workflow_schema.protocols['build_coordinates']

    build_coordinates.max_molecules = max_molecules
    build_coordinates.mass_density = mass_density

    density_workflow_schema.protocols['build_coordinates'] = build_coordinates.schema

    equilibration_simulation = simulation.RunOpenMMSimulation('')
    equilibration_simulation.schema = density_workflow_schema.protocols['equilibration_simulation']
    equilibration_simulation.ensemble = Ensemble.NVT
    equilibration_simulation.steps = equilibration_steps
    equilibration_simulation.output_frequency = equilibration_frequency

    density_workflow_schema.protocols['equilibration_simulation'] = equilibration_simulation.schema

    conditional_group = groups.ConditionalGroup('')
    conditional_group.schema = density_workflow_schema.protocols['conditional_group']

    conditional_group.protocols['production_simulation'].steps = production_steps
    conditional_group.protocols['production_simulation'].ensemble = Ensemble.NVT
    conditional_group.protocols['production_simulation'].output_frequency = production_frequency

    density_workflow_schema.protocols['conditional_group'] = conditional_group.schema

    return density_workflow_schema


def test_full_gradient_workflow():
    """Performs a complete test of the complete gradient workflow,
    including both the simulation and reweighting layers."""

    from propertyestimator.utils import setup_timestamp_logging
    setup_timestamp_logging()

    substance = Substance()
    substance.add_component(Substance.Component(smiles='O'),
                            Substance.MoleFraction())

    thermodynamic_state = ThermodynamicState(298*unit.kelvin)

    dummy_property = Density(thermodynamic_state,
                             PropertyPhase.Liquid,
                             substance,
                             1000.0*unit.kilogram / unit.meter**3,
                             1000.0*unit.kilogram / unit.meter**3,
                             source=CalculationSource('Dummy', {}))

    parameter_gradient_keys = [
        ParameterGradientKey(tag='Bonds', smirks='[#8:1]-[#1:2]', attribute='k')
    ]

    # Create a density workflow with debug settings.
    density_workflow_schema = create_debug_density_workflow(1, 0.001 * unit.grams / unit.milliliters, 50, 1, 100, 5)

    # Submit the estimate request using debug simulation. Set up the workflow options such
    # that the density estimation is guaranteed to succeed.
    options = PropertyEstimatorOptions()
    options.workflow_options = {
        'Density': {'SimulationLayer': WorkflowOptions(convergence_mode=WorkflowOptions.ConvergenceMode.NoChecks),
                    'ReweightingLayer': WorkflowOptions(convergence_mode=WorkflowOptions.ConvergenceMode.NoChecks)}
    }
    options.workflow_schemas = {'Density': {'SimulationLayer': density_workflow_schema}}

    with tempfile.TemporaryDirectory() as temporary_directory:

        force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

        storage_directory = path.join(temporary_directory, 'storage')
        working_directory = path.join(temporary_directory, 'working')

        dummy_data_set = PhysicalPropertyDataSet()
        dummy_data_set.properties[dummy_property.substance.identifier] = [dummy_property]

        calculation_backend = DaskLocalCluster(1, ComputeResources())
        storage_backend = LocalFileStorage(storage_directory)

        PropertyEstimatorServer(calculation_backend, storage_backend, 8001, working_directory)

        property_estimator = PropertyEstimatorClient(ConnectionOptions(server_port=8001))
        options.allowed_calculation_layers = ['SimulationLayer']

        request = property_estimator.request_estimate(dummy_data_set, force_field, options, parameter_gradient_keys)
        result = request.results(synchronous=True, polling_interval=0)

        assert not isinstance(result, PropertyEstimatorException)
        assert len(result.exceptions) == 0
        assert len(result.unsuccessful_properties) == 0
        assert len(result.queued_properties) == 0
        assert len(result.estimated_properties) == 1

        options.allowed_calculation_layers = ['ReweightingLayer']

        property_estimator = PropertyEstimatorClient(ConnectionOptions(server_port=8001))
        request = property_estimator.request_estimate(dummy_data_set, force_field, options, parameter_gradient_keys)
        result = request.results(synchronous=True, polling_interval=0)

        assert not isinstance(result, PropertyEstimatorException)
        assert len(result.exceptions) == 0
        assert len(result.unsuccessful_properties) == 0
        assert len(result.queued_properties) == 0
        assert len(result.estimated_properties) == 1


if __name__ == '__main__':
    test_full_gradient_workflow()
