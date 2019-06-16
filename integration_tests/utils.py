from propertyestimator.properties import Density
from propertyestimator.protocols import coordinates, groups, simulation


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
