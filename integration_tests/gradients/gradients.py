#!/usr/bin/env python
import os
import shutil

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import unit

from propertyestimator.backends import ComputeResources, DaskLocalCluster
from propertyestimator.layers.layers import CalculationLayerResult
from propertyestimator.properties import Density, PropertyPhase, CalculationSource
from propertyestimator.properties.properties import ParameterGradientKey
from propertyestimator.protocols.analysis import ExtractAverageStatistic, ExtractUncorrelatedTrajectoryData
from propertyestimator.protocols.gradients import GradientReducedPotentials, CentralDifferenceGradient
from propertyestimator.protocols.groups import ProtocolGroup
from propertyestimator.protocols.reweighting import ReweightWithMBARProtocol
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging, statistics
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow import WorkflowSchema, Workflow, WorkflowGraph
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


def find_differentiable_parameters(force_field, topology):
    """A generator function which yields all of the differentiable parameters
    which may be applied to a given topology.

    Parameters
    ----------
    force_field: openforcefield.typing.engines.smirnoff.ForceField
        The force field being applied.
    topology: openforcefield.topology.Topology
        The topology the force field is being applied to.

    Returns
    -------
    str
        The type of parameter, e.g. Bonds, Angles...
    ParameterType
        The differentiable parameter type.
    str
        The differentiable attribute (e.g. k or length) of the parameter.
    """

    parameters_by_tag_smirks = {}

    for parameter_set in force_field.label_molecules(topology):

        for parameter_tag in parameter_set:

            if parameter_tag in ['Electrostatics', 'ToolkitAM1BCC']:
                continue

            if parameter_tag not in parameters_by_tag_smirks:
                parameters_by_tag_smirks[parameter_tag] = {}

            for parameter in parameter_set[parameter_tag].store.values():
                parameters_by_tag_smirks[parameter_tag][parameter.smirks] = parameter

    parameter_keys = []

    for parameter_tag in parameters_by_tag_smirks:

        for smirks in parameters_by_tag_smirks[parameter_tag]:

            parameter = parameters_by_tag_smirks[parameter_tag][smirks]

            for parameter_attribute in parameter._REQUIRE_UNITS:

                if not hasattr(parameter, parameter_attribute):
                    continue

                parameter_value = getattr(parameter, parameter_attribute)

                if not isinstance(parameter_value, unit.Quantity):
                    continue

                parameter_value = parameter_value.value_in_unit_system(unit.md_unit_system)

                if not isinstance(parameter_value, float) and not isinstance(parameter_value, int):
                    continue

                parameter_keys.append(ParameterGradientKey(parameter_tag, parameter.smirks, parameter_attribute))

    return parameter_keys


def build_dummy_property(force_field, property_class=Density):

    methanol_molecule = Molecule.from_smiles('CO')
    methanol_topology = Topology.from_molecules(methanol_molecule)

    parameter_keys = find_differentiable_parameters(force_field, methanol_topology)

    extract_densities = ExtractAverageStatistic('extract_densities')
    extract_densities.bootstrap_iterations = 0
    extract_densities.statistics_path = 'methanol/methanol.csv'
    extract_densities.statistics_type = statistics.ObservableType.Density
    extract_densities.execute('', None)

    extract_trajectory = ExtractUncorrelatedTrajectoryData('extract_trajectory')
    extract_trajectory.input_coordinate_file = 'methanol/methanol.pdb'
    extract_trajectory.input_trajectory_path = 'methanol/methanol.dcd'
    extract_trajectory.equilibration_index = extract_densities.equilibration_index
    extract_trajectory.statistical_inefficiency = extract_densities.statistical_inefficiency
    extract_trajectory.execute('', None)

    substance = Substance()
    substance.add_component(Substance.Component('CO'), Substance.MoleFraction(1.0))

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram)

    dummy_property.source = CalculationSource(fidelity='dummy', provenance={})

    dummy_property.metadata = {
        'trajectory_file_path': extract_trajectory.output_trajectory_path,
        'coordinate_file_path': 'methanol/methanol.pdb',
        'observable_values': extract_densities.uncorrelated_values,
        'parameter_gradient_keys': parameter_keys
    }

    return dummy_property


def build_gradient_group(use_subset):

    reduced_potentials = GradientReducedPotentials('gradient_reduced_potentials_$(repl)')

    reduced_potentials.substance = ProtocolPath('substance', 'global')
    reduced_potentials.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
    reduced_potentials.reference_force_field_paths = [ProtocolPath('force_field_path', 'global')]
    reduced_potentials.force_field_path = ProtocolPath('force_field_path', 'global')

    reduced_potentials.trajectory_file_path = ProtocolPath('trajectory_file_path', 'global')
    reduced_potentials.coordinate_file_path = ProtocolPath('coordinate_file_path', 'global')

    reduced_potentials.parameter_key = ReplicatorValue('repl')
    reduced_potentials.perturbation_scale = 1.0e-4

    reduced_potentials.use_subset_of_force_field = use_subset

    reverse_mbar = ReweightWithMBARProtocol('reverse_mbar_$(repl)')
    reverse_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    reverse_mbar.reference_observables = [ProtocolPath('observable_values', 'global')]
    reverse_mbar.target_reduced_potentials = [ProtocolPath('reverse_potentials_path', reduced_potentials.id)]
    reverse_mbar.required_effective_samples = 0
    reverse_mbar.bootstrap_uncertainties = False

    forward_mbar = ReweightWithMBARProtocol('forward_mbar_$(repl)')
    forward_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    forward_mbar.reference_observables = [ProtocolPath('observable_values', 'global')]
    forward_mbar.target_reduced_potentials = [ProtocolPath('forward_potentials_path', reduced_potentials.id)]
    forward_mbar.required_effective_samples = 0
    forward_mbar.bootstrap_uncertainties = False

    central_difference = CentralDifferenceGradient('central_difference_$(repl)')
    central_difference.parameter_key = ReplicatorValue('repl')
    central_difference.reverse_observable_value = ProtocolPath('value', reverse_mbar.id)
    central_difference.forward_observable_value = ProtocolPath('value', forward_mbar.id)
    central_difference.reverse_parameter_value = ProtocolPath('reverse_parameter_value', reduced_potentials.id)
    central_difference.forward_parameter_value = ProtocolPath('forward_parameter_value', reduced_potentials.id)

    gradient_group = ProtocolGroup('gradient_group_$(repl)')
    gradient_group.add_protocols(reduced_potentials, reverse_mbar, forward_mbar, central_difference)

    return gradient_group.schema


def estimate_gradients(physical_property, metadata, use_subset, compute_backend):

    gradient_group_schema = build_gradient_group(use_subset)

    gradient_workflow_schema = WorkflowSchema('Density')
    gradient_workflow_schema.protocols[gradient_group_schema.id] = gradient_group_schema

    protocols_to_replicate = [ProtocolPath('', gradient_group_schema.id)]

    protocols_to_replicate.extend([ProtocolPath('', gradient_group_schema.id, protocol_schema.id) for
                                   protocol_schema in gradient_group_schema.grouped_protocol_schemas])

    parameter_replicator = ProtocolReplicator(replicator_id='repl')
    parameter_replicator.protocols_to_replicate = protocols_to_replicate
    parameter_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

    gradient_workflow_schema.replicators = [parameter_replicator]

    gradient_workflow_schema.gradients_sources = [ProtocolPath('gradient', gradient_group_schema.id,
                                                                           'central_difference_$(repl)')]

    gradient_workflow = Workflow(physical_property, metadata)
    gradient_workflow.schema = gradient_workflow_schema

    if os.path.isdir('working_directory'):
        shutil.rmtree('working_directory')

    os.makedirs('working_directory')

    workflow_graph = WorkflowGraph('working_directory')
    workflow_graph.add_workflow(gradient_workflow)

    futures = workflow_graph.submit(compute_backend, False)
    exceptions = []

    estimated_properties = []

    for future in futures:

        result = future.result()

        if isinstance(result, PropertyEstimatorException):

            exceptions.append(result)
            continue

        elif isinstance(result, CalculationLayerResult) and result.exception is not None:

            exceptions.append(result.exception)
            continue

        estimated_properties.append(result.calculated_property)

    if len(exceptions) > 0:
        error_message = '\n'.join([f'{exception.directory}: {exception.message}' for exception in exceptions])
        raise RuntimeError(error_message)

    return estimated_properties


def main():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""
    setup_timestamp_logging()

    # Create the backend which the gradients will be estimated on.
    # compute_resource = ComputeResources(number_of_threads=1, number_of_gpus=1,
    #                                     preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

    compute_resource = ComputeResources(number_of_threads=1)

    compute_backend = DaskLocalCluster(number_of_workers=1,
                                       resources_per_worker=compute_resource)

    compute_backend.start()

    # Load in the force field
    force_field_path = build_tip3p_smirnoff_force_field()
    force_field = ForceField(force_field_path)

    # Setup the dummy methanol property which will act as a test case
    # for gradient computation.
    methanol_property = build_dummy_property(force_field)
    methanol_metadata = Workflow.generate_default_metadata(methanol_property,
                                                           force_field_path)

    # Compute the gradients using only a subset of the force field.
    estimated_properties = estimate_gradients(methanol_property,
                                              methanol_metadata,
                                              True,
                                              compute_backend)

    fast_gradients = {}

    for estimated_property in estimated_properties:
        for gradient in estimated_property.gradients:
            fast_gradients[gradient.key] = gradient.value

    # Compute the gradients using the full force field.
    estimated_properties = estimate_gradients(methanol_property,
                                              methanol_metadata,
                                              False,
                                              compute_backend)

    slow_gradients = {}

    for estimated_property in estimated_properties:
        for gradient in estimated_property.gradients:
            slow_gradients[gradient.key] = gradient.value

    assert len(slow_gradients) == len(fast_gradients)

    for parameter_key in fast_gradients:

        gradient_slow = slow_gradients[parameter_key]
        gradient_fast = fast_gradients[parameter_key]

        absolute_difference = gradient_fast - gradient_slow
        relative_difference = (gradient_fast - gradient_slow) / abs(gradient_slow)

        print(f'{parameter_key.tag}: {parameter_key.smirks} - dE/{parameter_key.attribute} '
              f'slow={gradient_slow} '
              f'fast={gradient_fast} '
              f'absdiff={absolute_difference} '
              f'reldiff={relative_difference}')

    compute_backend.stop()


if __name__ == "__main__":
    main()
