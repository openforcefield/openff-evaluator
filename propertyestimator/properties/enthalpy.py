"""
A collection of enthalpy physical property definitions.
"""

from collections import namedtuple

import numpy as np

from propertyestimator import unit
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.properties.properties import PhysicalProperty, PropertyPhase, ParameterGradientKey, \
    ParameterGradient
from propertyestimator.protocols import analysis, coordinates, forcefield, groups, miscellaneous, simulation, \
    reweighting, gradients
from propertyestimator.protocols.utils import generate_base_reweighting_protocols, generate_base_simulation_protocols, \
    generate_gradient_protocol_group
from propertyestimator.storage import StoredSimulationData
from propertyestimator.storage.dataclasses import StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType, StatisticsArray
from propertyestimator.workflow import plugins, WorkflowOptions
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.workflow.schemas import ProtocolReplicator, WorkflowSimulationDataToStore, WorkflowSchema, \
    WorkflowDataCollectionToStore
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


@plugins.register_calculation_protocol()
class WeightValueByMoleFraction(BaseProtocol):
    """Multiplies a value by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(EstimatedQuantity)
    def value(self, value):
        """The value to be weighted."""
        pass

    @protocol_input(Substance)
    def component(self, value):
        """The component (e.g water) to which this value belongs."""
        pass

    @protocol_input(Substance)
    def full_substance(self, value):
        """The full substance of which the component of interest is a part."""
        pass

    @protocol_output(EstimatedQuantity)
    def weighted_value(self, value):
        """The value weighted by the `component`s mole fraction as determined from
        the `full_substance`."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._value = None
        self._component = None
        self._full_substance = None

        self._weighted_value = None

    def execute(self, directory, available_resources):

        assert len(self._component.components) == 1

        main_component = self._component.components[0]
        amount = self._full_substance.get_amount(main_component)

        if not isinstance(amount, Substance.MoleFraction):

            return PropertyEstimatorException(directory=directory,
                                              message=f'The component {main_component} was given in an '
                                                      f'exact amount, and not a mole fraction')

        self._weighted_value = self.value * amount.value
        return self._get_output_dictionary()


@plugins.register_calculation_protocol()
class EnthalpyFluctuationGradient(BaseProtocol):
    """Multiplies a value by the mole fraction of a component
    in a mixture substance.
    """

    @protocol_input(ParameterGradientKey)
    def parameter_key(self):
        """The key that describes which parameters this
        gradient was estimated for."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic conditions under which the enthalpy was measured."""
        pass

    @protocol_input(str)
    def statistics_path(self):
        """The path to the values of U(theta) and u(theta) stored in a `StatisticsArray`."""
        pass

    @protocol_input(str)
    def reverse_statistics_path(self):
        """The path to the values of U(theta-h) and u(theta-h) stored in a `StatisticsArray`."""
        pass

    @protocol_input(str)
    def forward_statistics_path(self):
        """The path to the values of U(theta+h) and u(theta+h) stored in a `StatisticsArray`."""
        pass

    @protocol_input(unit.Quantity)
    def reverse_parameter_value(self):
        """The value of theta-h."""
        pass

    @protocol_input(unit.Quantity)
    def forward_parameter_value(self):
        """The value of theta+h."""
        pass

    @protocol_output(ParameterGradient)
    def gradient(self):
        """The estimated gradient."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._parameter_key = None
        self._thermodynamic_state = None

        self._statistics_path = None

        self._reverse_statistics_path = None
        self._forward_statistics_path = None

        self._reverse_parameter_value = None
        self._forward_parameter_value = None

        self._gradient = None

    def execute(self, directory, available_resources):

        reverse_statistics = StatisticsArray.from_pandas_csv(self._reverse_statistics_path)
        forward_statistics = StatisticsArray.from_pandas_csv(self._forward_statistics_path)

        potential_energy_gradients = ((forward_statistics[ObservableType.PotentialEnergy] -
                                       reverse_statistics[ObservableType.PotentialEnergy]) /
                                      (self._forward_parameter_value - self._reverse_parameter_value))

        statistics = StatisticsArray.from_pandas_csv(self._statistics_path)
        potential_energies = statistics[ObservableType.PotentialEnergy]

        average_energy_gradient = np.mean(potential_energy_gradients)
        average_energy = np.mean(potential_energies)

        beta = 1.0 / (self._thermodynamic_state.temperature * unit.molar_gas_constant)
        beta.ito(unit.mole / unit.kilojoule)

        gradient_value = (average_energy_gradient -
                          beta * (np.mean(potential_energies * potential_energy_gradients) -
                                  average_energy * average_energy_gradient))

        self._gradient = ParameterGradient(self._parameter_key, gradient_value)

        return self._get_output_dictionary()


@register_estimable_property()
@register_thermoml_property(thermoml_string='Excess molar enthalpy (molar enthalpy of mixing), kJ/mol',
                            supported_phases=PropertyPhase.Liquid)
class EnthalpyOfMixing(PhysicalProperty):
    """A class representation of an enthalpy of mixing property"""

    EnthalpyWorkflow = namedtuple('EnthalpySchema', 'build_coordinates '
                                                    'assign_topology '
                                                    'energy_minimisation '
                                                    'npt_equilibration '
                                                    'converge_uncertainty '
                                                    'subsample_trajectory '
                                                    'subsample_statistics ')

    @property
    def multi_component_property(self):
        return True

    @property
    def required_data_class(self):
        return StoredSimulationData

    @staticmethod
    def _get_enthalpy_workflow(id_prefix='', weight_by_mole_fraction=False, options=None, substance_reference=None):
        """Returns the set of protocols which when combined in a workflow
        will yield the enthalpy of a substance.

        Parameters
        ----------
        id_prefix: str
            A prefix to append to the id of each of the returned protocols.
        weight_by_mole_fraction: bool
            If true, an extra protocol will be added to weight the calculated
            enthalpy by the mole fraction of the component inside of the
            convergence loop.
        options: WorkflowOptions
            The options to use when setting up the workflows.
        substance_reference: ProtocolPath, optional
            An optional protocol path (or replicator reference) to the substance
            whose enthalpy is being estimated.

        Returns
        -------
        EnthalpyOfMixing.EnthalpyWorkflow
            The protocols used to estimate the enthalpy of a substance.
        """

        if substance_reference is None:
            substance_reference = ProtocolPath('substance', 'global')

        build_coordinates = coordinates.BuildCoordinatesPackmol(id_prefix + 'build_coordinates')

        build_coordinates.substance = substance_reference

        assign_topology = forcefield.BuildSmirnoffSystem(id_prefix + 'build_topology')

        assign_topology.force_field_path = ProtocolPath('force_field_path', 'global')

        assign_topology.coordinate_file_path = ProtocolPath('coordinate_file_path', build_coordinates.id)
        assign_topology.substance = substance_reference

        # Equilibration
        energy_minimisation = simulation.RunEnergyMinimisation(id_prefix + 'energy_minimisation')

        energy_minimisation.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
        energy_minimisation.system_path = ProtocolPath('system_path', assign_topology.id)

        npt_equilibration = simulation.RunOpenMMSimulation(id_prefix + 'npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 100000  # Debug settings.
        npt_equilibration.output_frequency = 5000  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system_path = ProtocolPath('system_path', assign_topology.id)

        # Production
        npt_production = simulation.RunOpenMMSimulation(id_prefix + 'npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 500000  # Debug settings.
        npt_production.output_frequency = 5000  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system_path = ProtocolPath('system_path', assign_topology.id)

        # Analysis
        extract_enthalpy = analysis.ExtractAverageStatistic(id_prefix + 'extract_enthalpy')

        extract_enthalpy.statistics_type = ObservableType.Enthalpy
        extract_enthalpy.statistics_path = ProtocolPath('statistics_file_path', npt_production.id)

        # Set up a conditional group to ensure convergence of uncertainty
        converge_uncertainty = groups.ConditionalGroup(id_prefix + 'converge_uncertainty')
        converge_uncertainty.add_protocols(npt_production, extract_enthalpy)

        # Divide by the number of molecules in the system
        divide_by_molecules = miscellaneous.DivideValue(id_prefix + 'divide_by_molecules')
        divide_by_molecules.value = ProtocolPath('value', extract_enthalpy.id)
        divide_by_molecules.divisor = ProtocolPath('final_number_of_molecules', build_coordinates.id)

        converge_uncertainty.add_protocols(divide_by_molecules)

        if weight_by_mole_fraction:

            # The component workflows need an extra step to multiply their enthalpies by their
            # relative mole fraction.
            weight_by_mole_fraction = WeightValueByMoleFraction(id_prefix + 'weight_by_mole_fraction')

            weight_by_mole_fraction.value = ProtocolPath('value', extract_enthalpy.id)
            weight_by_mole_fraction.full_substance = ProtocolPath('substance', 'global')

            # Again, set the component as a placeholder which will be set by the replicator.
            weight_by_mole_fraction.component = ReplicatorValue('repl')

            converge_uncertainty.add_protocols(weight_by_mole_fraction)

            # Make sure to divide the weighted value.
            divide_by_molecules.value = ProtocolPath('weighted_value', weight_by_mole_fraction.id)

        converge_uncertainty.max_iterations = 150

        if options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks:

            condition = groups.ConditionalGroup.Condition()

            condition.left_hand_value = ProtocolPath('result.uncertainty', converge_uncertainty.id,
                                                                           divide_by_molecules.id)

            condition.right_hand_value = ProtocolPath('per_component_uncertainty', 'global')
            condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

            converge_uncertainty.add_condition(condition)

        # Set up the storage protocols.
        statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                converge_uncertainty.id, extract_enthalpy.id)

        equilibration_index = ProtocolPath('equilibration_index',
                                           converge_uncertainty.id, extract_enthalpy.id)

        # Extract the uncorrelated trajectory.
        extract_uncorrelated_trajectory = analysis.ExtractUncorrelatedTrajectoryData(id_prefix + 'extract_traj')

        extract_uncorrelated_trajectory.statistical_inefficiency = statistical_inefficiency
        extract_uncorrelated_trajectory.equilibration_index = equilibration_index

        extract_uncorrelated_trajectory.input_coordinate_file = ProtocolPath('output_coordinate_file',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        extract_uncorrelated_trajectory.input_trajectory_path = ProtocolPath('trajectory_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)

        # Extract the uncorrelated statistics.
        extract_uncorrelated_statistics = analysis.ExtractUncorrelatedStatisticsData(id_prefix + 'extract_stats')

        extract_uncorrelated_statistics.statistical_inefficiency = statistical_inefficiency
        extract_uncorrelated_statistics.equilibration_index = equilibration_index

        extract_uncorrelated_statistics.input_statistics_path = ProtocolPath('statistics_file_path',
                                                                             converge_uncertainty.id,
                                                                             npt_production.id)
        # # Set up the gradient calculations
        # gradient_group, gradient_replicator, gradient_source = \
        #     EnthalpyOfMixing._get_gradient_protocols(id_prefix,
        #                                              weight_by_mole_fraction,
        #                                              substance_reference,
        #                                              ProtocolPath('final_number_of_molecules',
        #                                                           build_coordinates.id),
        #                                              ProtocolPath('output_coordinate_file', npt_equilibration.id),
        #                                              ProtocolPath('output_trajectory_path',
        #                                                           extract_uncorrelated_trajectory.id),
        #                                              ProtocolPath('uncorrelated_values',
        #                                                           converge_uncertainty.id,
        #                                                           extract_enthalpy.id))

        # noinspection PyCallByClass
        return EnthalpyOfMixing.EnthalpyWorkflow(build_coordinates, assign_topology,
                                                 energy_minimisation, npt_equilibration,
                                                 converge_uncertainty, extract_uncorrelated_trajectory,
                                                 extract_uncorrelated_statistics)

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return EnthalpyOfMixing.get_default_simulation_workflow_schema(options)
        elif calculation_layer == 'ReweightingLayer':
            return EnthalpyOfMixing.get_default_reweighting_workflow_schema(options)

        return None

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        from direct simulations.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        # Set up a general workflow for calculating the enthalpy of one of the system components.
        # Here we affix a prefix which contains the special string $(comp_index). Protocols which are
        # replicated by a replicator will have the $(comp_index) tag in their id replaced by the index
        # of the replication.
        component_workflow = EnthalpyOfMixing._get_enthalpy_workflow('component_$(repl)_', True, options)

        # Set the substance of the build_coordinates and assign_topology protocols
        # as a placeholder for now - these will be later set by the replicator.
        component_workflow.build_coordinates.substance = ReplicatorValue('repl')
        component_workflow.assign_topology.substance = ReplicatorValue('repl')

        # Set up a workflow to calculate the enthalpy of the full, mixed system.
        mixed_system_workflow = EnthalpyOfMixing._get_enthalpy_workflow('mixed_', False, options)

        # Finally, set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the mixed system enthalpy.
        add_component_enthalpies = miscellaneous.AddValues('add_component_enthalpies')

        # Although we only give a list of a single ProtocolPath pointing to our template
        # component workflow's `weight_by_mole_fraction` protocol, the replicator
        # will actually populate this list with references to all of the newly generated
        # protocols of the individual components.
        add_component_enthalpies.values = [ProtocolPath('result', component_workflow.converge_uncertainty.id,
                                                                  'component_$(repl)_divide_by_molecules')]

        schema.protocols[add_component_enthalpies.id] = add_component_enthalpies.schema

        calculate_enthalpy_of_mixing = miscellaneous.SubtractValues('calculate_enthalpy_of_mixing')

        calculate_enthalpy_of_mixing.value_b = ProtocolPath('result', mixed_system_workflow.converge_uncertainty.id,
                                                                     'mixed_divide_by_molecules')
        calculate_enthalpy_of_mixing.value_a = ProtocolPath('result', add_component_enthalpies.id)

        schema.protocols[calculate_enthalpy_of_mixing.id] = calculate_enthalpy_of_mixing.schema

        for component_protocol in component_workflow:
            schema.protocols[component_protocol.id] = component_protocol.schema

        for mixed_protocol in mixed_system_workflow:
            schema.protocols[mixed_protocol.id] = mixed_protocol.schema

        # Create the replicator object which defines how the pure component
        # enthalpy estimation workflow will be replicated for each component.
        component_replicator = ProtocolReplicator(replicator_id='repl')

        component_replicator.protocols_to_replicate = []

        # Pass it paths to the protocols to be replicated.
        for component_protocol in component_workflow:
            component_replicator.protocols_to_replicate.append(ProtocolPath('', component_protocol.id))

        for component_protocol_id in component_workflow.converge_uncertainty.protocols:

            path_to_protocol = ProtocolPath('', component_workflow.converge_uncertainty.id,
                                                component_protocol_id)

            component_replicator.protocols_to_replicate.append(path_to_protocol)

        # Tell the replicator to take the components of a properties substance,
        # and pass these to the replicated workflows being produced, and in particular,
        # the inputs specified by the `template_targets`
        component_replicator.template_values = ProtocolPath('components', 'global')

        schema.replicators = [component_replicator]

        # Finally, tell the schemas where to look for its final values.
        schema.final_value_source = ProtocolPath('result', calculate_enthalpy_of_mixing.id)

        mixed_output_to_store = WorkflowSimulationDataToStore()

        mixed_output_to_store.total_number_of_molecules = ProtocolPath('final_number_of_molecules',
                                                                       mixed_system_workflow.build_coordinates.id)

        mixed_output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                  mixed_system_workflow.subsample_trajectory.id)

        mixed_output_to_store.coordinate_file_path = ProtocolPath('output_coordinate_file',
                                                                  mixed_system_workflow.converge_uncertainty.id,
                                                                  'mixed_npt_production')

        mixed_output_to_store.statistics_file_path = ProtocolPath('output_statistics_path',
                                                                  mixed_system_workflow.subsample_statistics.id)

        mixed_output_to_store.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                      mixed_system_workflow.converge_uncertainty.id,
                                                                      'mixed_extract_enthalpy')

        component_output_to_store = WorkflowSimulationDataToStore()

        component_output_to_store.substance = ReplicatorValue('repl')

        component_output_to_store.total_number_of_molecules = ProtocolPath('final_number_of_molecules',
                                                                           component_workflow.build_coordinates.id)

        component_output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                      component_workflow.subsample_trajectory.id)

        component_output_to_store.coordinate_file_path = ProtocolPath('output_coordinate_file',
                                                                      component_workflow.converge_uncertainty.id,
                                                                      'component_$(repl)_npt_production')

        component_output_to_store.statistics_file_path = ProtocolPath('output_statistics_path',
                                                                      component_workflow.subsample_statistics.id)

        component_output_to_store.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                          component_workflow.converge_uncertainty.id,
                                                                          'component_$(repl)_extract_enthalpy')

        schema.outputs_to_store = {
            'mixed_system': mixed_output_to_store,
            'component_$(repl)': component_output_to_store
        }

        return schema

    @staticmethod
    def get_default_reweighting_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        by reweighting existing data.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Set up the protocols which will reweight data for the full system.
        extract_mixed_enthalpy = analysis.ExtractAverageStatistic('extract_enthalpy_$(mix_data_repl)_mixture')
        extract_mixed_enthalpy.statistics_type = ObservableType.Enthalpy

        mixture_protocols, mixture_data_replicator = generate_base_reweighting_protocols(extract_mixed_enthalpy,
                                                                                         options,
                                                                                         'mix_data_repl',
                                                                                         '_mixture')

        extract_mixed_enthalpy.statistics_path = ProtocolPath('statistics_file_path',
                                                              mixture_protocols.unpack_stored_data.id)

        # Set up the protocols which will reweight data for each of the components.
        extract_pure_enthalpy = analysis.ExtractAverageStatistic(
            'extract_enthalpy_$(pure_data_repl)_comp_$(comp_repl)')
        extract_pure_enthalpy.statistics_type = ObservableType.Enthalpy

        pure_protocols, pure_data_replicator = generate_base_reweighting_protocols(extract_pure_enthalpy,
                                                                                   options,
                                                                                   'pure_data_repl',
                                                                                   '_pure_$(comp_repl)')

        extract_pure_enthalpy.statistics_path = ProtocolPath('statistics_file_path',
                                                             pure_protocols.unpack_stored_data.id)

        # Make sure the replicator is only replicating over data from the pure component.
        pure_data_replicator.template_values = ProtocolPath('component_data[$(comp_repl)]', 'global')

        # Set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the mixed system enthalpy.
        weight_by_mole_fraction = WeightValueByMoleFraction('weight_comp_$(comp_repl)')
        weight_by_mole_fraction.value = ProtocolPath('value', pure_protocols.mbar_protocol.id)
        weight_by_mole_fraction.full_substance = ProtocolPath('substance', 'global')
        weight_by_mole_fraction.component = ReplicatorValue('comp_repl')

        # Divide by the component enthalpies by the number of molecules in the system
        # TODO cleanup replicators so can target a single replicated protocol rather
        #      than a list if possible
        divisor_data = reweighting.UnpackStoredSimulationData('divisor_data')
        divisor_data.simulation_data_path = ProtocolPath('full_system_data[0]', 'global')

        pure_divide_by_molecules = miscellaneous.DivideValue('divide_by_molecules_$(comp_repl)')
        pure_divide_by_molecules.value = ProtocolPath('weighted_value', weight_by_mole_fraction.id)
        pure_divide_by_molecules.divisor = ProtocolPath('total_number_of_molecules', divisor_data.id)

        # Divide by the mixture enthalpy by the number of molecules in the system
        mixture_divide_by_molecules = miscellaneous.DivideValue('divide_by_mixture_molecules')
        mixture_divide_by_molecules.value = ProtocolPath('value', mixture_protocols.mbar_protocol.id)
        mixture_divide_by_molecules.divisor = ProtocolPath('total_number_of_molecules', divisor_data.id)

        add_component_enthalpies = miscellaneous.AddValues('add_component_enthalpies')
        add_component_enthalpies.values = [ProtocolPath('result', pure_divide_by_molecules.id)]

        calculate_enthalpy_of_mixing = miscellaneous.SubtractValues('calculate_enthalpy_of_mixing')
        calculate_enthalpy_of_mixing.value_b = ProtocolPath('result', mixture_divide_by_molecules.id)
        calculate_enthalpy_of_mixing.value_a = ProtocolPath('result', add_component_enthalpies.id)

        # Set up a replicator that will re-run the pure reweighting workflow for each
        # component in the system.
        pure_component_replicator = ProtocolReplicator(replicator_id='comp_repl')
        pure_component_replicator.protocols_to_replicate = [
            ProtocolPath('', weight_by_mole_fraction.id),
            ProtocolPath('', pure_divide_by_molecules.id)
        ]

        for pure_protocol in pure_protocols:
            pure_component_replicator.protocols_to_replicate.append(ProtocolPath('', pure_protocol.id))

        pure_component_replicator.template_values = ProtocolPath('components', 'global')

        # Build the final workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        schema.protocols = dict()

        schema.protocols[divisor_data.id] = divisor_data.schema

        schema.protocols.update({protocol.id: protocol.schema for protocol in mixture_protocols})
        schema.protocols.update({protocol.id: protocol.schema for protocol in pure_protocols})

        schema.protocols[weight_by_mole_fraction.id] = weight_by_mole_fraction.schema
        schema.protocols[pure_divide_by_molecules.id] = pure_divide_by_molecules.schema
        schema.protocols[mixture_divide_by_molecules.id] = mixture_divide_by_molecules.schema
        schema.protocols[add_component_enthalpies.id] = add_component_enthalpies.schema
        schema.protocols[calculate_enthalpy_of_mixing.id] = calculate_enthalpy_of_mixing.schema

        schema.replicators = [
            mixture_data_replicator,
            pure_component_replicator,
            pure_data_replicator
        ]

        schema.final_value_source = ProtocolPath('result', calculate_enthalpy_of_mixing.id)

        return schema


@register_estimable_property()
@register_thermoml_property('Molar enthalpy of vaporization or sublimation, kJ/mol',
                            supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas)
class EnthalpyOfVaporization(PhysicalProperty):
    """A class representation of an enthalpy of vaporization property"""

    @property
    def multi_component_property(self):
        """Returns whether this property is dependant on properties of the
        full mixed substance, or whether it is also dependant on the properties
        of the individual components also.
        """
        return False

    @property
    def required_data_class(self):
        return StoredDataCollection

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return EnthalpyOfVaporization.get_default_simulation_workflow_schema(options)
        # elif calculation_layer == 'ReweightingLayer':
        #     return EnthalpyOfVaporization.get_default_reweighting_workflow_schema(options)

        return None

    @staticmethod
    def _generate_gradient_group(reference_force_field_paths,
                                 target_force_field_path,
                                 coordinate_file_path,
                                 trajectory_file_path,
                                 statistics_file_path,
                                 replicator_id='repl',
                                 perturbation_scale=1.0e-3,
                                 id_prefix='',
                                 enable_pbc=True):

        # Define the protocol which will evaluate the reduced potentials of the
        # reference, forward and reverse states using only a subset of the full
        # force field.
        reduced_potentials = gradients.GradientReducedPotentials(f'{id_prefix}gradient_reduced_potentials_'
                                                                 f'$({replicator_id})')

        reduced_potentials.substance = ProtocolPath('substance', 'global')
        reduced_potentials.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
        reduced_potentials.reference_force_field_paths = reference_force_field_paths
        reduced_potentials.force_field_path = target_force_field_path
        reduced_potentials.trajectory_file_path = trajectory_file_path
        reduced_potentials.coordinate_file_path = coordinate_file_path
        reduced_potentials.parameter_key = ReplicatorValue(replicator_id)
        reduced_potentials.perturbation_scale = perturbation_scale
        reduced_potentials.use_subset_of_force_field = True
        reduced_potentials.enable_pbc = enable_pbc

        # Set up the protocol which will actually evaluate the parameter gradient
        # using the fluctuation method.
        fluctuation_gradient = EnthalpyFluctuationGradient(f'{id_prefix}enthalpy_fluctuation_$({replicator_id})')
        fluctuation_gradient.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
        fluctuation_gradient.parameter_key = ReplicatorValue(replicator_id)
        fluctuation_gradient.statistics_path = statistics_file_path
        fluctuation_gradient.reverse_statistics_path = ProtocolPath('reverse_potentials_path', reduced_potentials.id)
        fluctuation_gradient.forward_statistics_path = ProtocolPath('forward_potentials_path', reduced_potentials.id)
        fluctuation_gradient.reverse_parameter_value = ProtocolPath('reverse_parameter_value', reduced_potentials.id)
        fluctuation_gradient.forward_parameter_value = ProtocolPath('forward_parameter_value', reduced_potentials.id)

        # Assemble all of the protocols into a convenient group wrapper.
        gradient_group = groups.ProtocolGroup(f'{id_prefix}gradient_group_$({replicator_id})')
        gradient_group.add_protocols(reduced_potentials, fluctuation_gradient)

        protocols_to_replicate = [ProtocolPath('', gradient_group.id)]

        protocols_to_replicate.extend([ProtocolPath('', gradient_group.id, protocol_id) for
                                       protocol_id in gradient_group.protocols])

        # Create the replicator which will copy the group for each parameter gradient
        # which will be calculated.
        parameter_replicator = ProtocolReplicator(replicator_id=replicator_id)
        parameter_replicator.protocols_to_replicate = protocols_to_replicate
        parameter_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

        return gradient_group, parameter_replicator, ProtocolPath('gradient', 
                                                                  gradient_group.id, 
                                                                  fluctuation_gradient.id)

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        from direct simulations.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Define the number of molecules for the liquid phase
        number_of_liquid_molecules = 1000

        # Define a custom conditional group.
        converge_uncertainty = groups.ConditionalGroup(f'converge_uncertainty')
        converge_uncertainty.max_iterations = 100

        # Define the protocols to perform the simulation in the liquid phase.
        extract_liquid_energy = analysis.ExtractAverageStatistic('extract_liquid_energy')
        extract_liquid_energy.statistics_type = ObservableType.PotentialEnergy
        extract_liquid_energy.divisor = number_of_liquid_molecules

        liquid_protocols, liquid_value_source, liquid_output_to_store = \
            generate_base_simulation_protocols(extract_liquid_energy, options, '_liquid', converge_uncertainty)

        # Make sure the number of molecules in the liquid is consistent.
        liquid_protocols.build_coordinates.max_molecules = number_of_liquid_molecules

        # Define the protocols to perform the simulation in the gas phase.
        extract_gas_energy = analysis.ExtractAverageStatistic('extract_gas_energy')
        extract_gas_energy.statistics_type = ObservableType.PotentialEnergy

        gas_protocols, gas_value_source, gas_output_to_store = \
            generate_base_simulation_protocols(extract_gas_energy, options, '_gas', converge_uncertainty)

        # Create only a single molecule in vacuum
        gas_protocols.build_coordinates.max_molecules = 1

        # Run the gas phase simulations in the NVT ensemble
        gas_protocols.energy_minimisation.enable_pbc = False

        gas_protocols.equilibration_simulation.ensemble = Ensemble.NVT
        gas_protocols.equilibration_simulation.enable_pbc = False
        gas_protocols.equilibration_simulation.save_rolling_statistics = False

        gas_protocols.production_simulation.ensemble = Ensemble.NVT
        gas_protocols.production_simulation.steps = 15000000
        gas_protocols.production_simulation.output_frequency = 5000
        gas_protocols.production_simulation.enable_pbc = False
        gas_protocols.production_simulation.save_rolling_statistics = False

        # Due to a bizarre issue where the OMM Reference platform is
        # the fastest at computing properties of a single molecule
        # in vacuum, we enforce those inputs which will force the
        # gas calculations to run on the Reference platform.
        gas_protocols.equilibration_simulation.high_precision = True
        gas_protocols.equilibration_simulation.allow_gpu_platforms = False
        gas_protocols.production_simulation.high_precision = True
        gas_protocols.production_simulation.allow_gpu_platforms = False

        # Combine the values to estimate the final energy of vaporization
        energy_of_vaporization = miscellaneous.SubtractValues('energy_of_vaporization')
        energy_of_vaporization.value_b = ProtocolPath('value', extract_gas_energy.id)
        energy_of_vaporization.value_a = ProtocolPath('value', extract_liquid_energy.id)

        ideal_volume = miscellaneous.MultiplyValue('ideal_volume')
        ideal_volume.value = EstimatedQuantity(1.0 * unit.molar_gas_constant,
                                               0.0 * unit.joule / unit.mole / unit.kelvin,
                                               'Universal Constant')
        ideal_volume.multiplier = ProtocolPath('thermodynamic_state.temperature', 'global')

        enthalpy_of_vaporization = miscellaneous.AddValues('enthalpy_of_vaporization')
        enthalpy_of_vaporization.values = [
            ProtocolPath('result', energy_of_vaporization.id),
            ProtocolPath('result', ideal_volume.id)
        ]

        # Add the extra protocols and conditions to the custom group.
        converge_uncertainty.add_protocols(energy_of_vaporization,
                                           ideal_volume,
                                           enthalpy_of_vaporization)

        if options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks:

            condition = groups.ConditionalGroup.Condition()
            condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

            condition.left_hand_value = ProtocolPath('result.uncertainty', converge_uncertainty.id,
                                                                           enthalpy_of_vaporization.id)
            condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

            converge_uncertainty.add_condition(condition)

        # Set up the liquid gradient calculations
        liquid_coordinate_source = ProtocolPath('output_coordinate_file', liquid_protocols.equilibration_simulation.id)
        liquid_trajectory_source = ProtocolPath('trajectory_file_path',
                                                liquid_protocols.converge_uncertainty.id,
                                                liquid_protocols.production_simulation.id)
        liquid_statistics_source = ProtocolPath('statistics_file_path',
                                                liquid_protocols.converge_uncertainty.id,
                                                liquid_protocols.production_simulation.id)

        liquid_gradient_group, liquid_gradient_replicator, liquid_gradient_source = EnthalpyOfVaporization.\
            _generate_gradient_group([ProtocolPath('force_field_path', 'global')],
                                      ProtocolPath('force_field_path', 'global'),
                                      liquid_coordinate_source,
                                      liquid_trajectory_source,
                                      liquid_statistics_source,
                                      id_prefix='liquid_')

        # Set up the gas gradient calculations
        gas_coordinate_source = ProtocolPath('output_coordinate_file', gas_protocols.equilibration_simulation.id)
        gas_trajectory_source = ProtocolPath('trajectory_file_path',
                                             gas_protocols.converge_uncertainty.id,
                                             gas_protocols.production_simulation.id)
        gas_statistics_source = ProtocolPath('statistics_file_path',
                                             gas_protocols.converge_uncertainty.id,
                                             gas_protocols.production_simulation.id)

        gas_gradient_group, gas_gradient_replicator, gas_gradient_source = EnthalpyOfVaporization. \
            _generate_gradient_group([ProtocolPath('force_field_path', 'global')],
                                     ProtocolPath('force_field_path', 'global'),
                                     gas_coordinate_source,
                                     gas_trajectory_source,
                                     gas_statistics_source,
                                     id_prefix='gas_',
                                     enable_pbc=False)

        # Combine the gradients.
        scale_liquid_gradient = gradients.DivideGradientByScalar('scale_liquid_gradient_$(repl)')
        scale_liquid_gradient.value = liquid_gradient_source
        scale_liquid_gradient.divisor = number_of_liquid_molecules

        combine_gradients = gradients.SubtractGradients('combine_gradients_$(repl)')
        combine_gradients.value_b = gas_gradient_source
        combine_gradients.value_a = ProtocolPath('result', scale_liquid_gradient.id)

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(replicator_id=liquid_gradient_replicator.id)
        gradient_replicator.protocols_to_replicate = [*liquid_gradient_replicator.protocols_to_replicate,
                                                      *gas_gradient_replicator.protocols_to_replicate,
                                                      ProtocolPath('', scale_liquid_gradient.id),
                                                      ProtocolPath('', combine_gradients.id)]
        gradient_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfVaporization.__name__)
        schema.id = '{}{}'.format(EnthalpyOfVaporization.__name__, 'Schema')

        schema.protocols = {
            liquid_protocols.build_coordinates.id: liquid_protocols.build_coordinates.schema,
            liquid_protocols.assign_parameters.id: liquid_protocols.assign_parameters.schema,
            liquid_protocols.energy_minimisation.id: liquid_protocols.energy_minimisation.schema,
            liquid_protocols.equilibration_simulation.id: liquid_protocols.equilibration_simulation.schema,

            gas_protocols.build_coordinates.id: gas_protocols.build_coordinates.schema,
            gas_protocols.assign_parameters.id: gas_protocols.assign_parameters.schema,
            gas_protocols.energy_minimisation.id: gas_protocols.energy_minimisation.schema,
            gas_protocols.equilibration_simulation.id: gas_protocols.equilibration_simulation.schema,

            converge_uncertainty.id: converge_uncertainty.schema,

            liquid_protocols.extract_uncorrelated_trajectory.id:
                liquid_protocols.extract_uncorrelated_trajectory.schema,
            liquid_protocols.extract_uncorrelated_statistics.id:
                liquid_protocols.extract_uncorrelated_statistics.schema,

            gas_protocols.extract_uncorrelated_trajectory.id: gas_protocols.extract_uncorrelated_trajectory.schema,
            gas_protocols.extract_uncorrelated_statistics.id: gas_protocols.extract_uncorrelated_statistics.schema,

            liquid_gradient_group.id: liquid_gradient_group.schema,
            gas_gradient_group.id: gas_gradient_group.schema,

            scale_liquid_gradient.id: scale_liquid_gradient.schema,
            combine_gradients.id: combine_gradients.schema
        }

        schema.replicators = [gradient_replicator]

        data_to_store = WorkflowDataCollectionToStore()

        data_to_store.data['liquid'] = liquid_output_to_store
        data_to_store.data['gas'] = gas_output_to_store

        schema.outputs_to_store = {'full_system_data': data_to_store}

        schema.gradients_sources = [ProtocolPath('result', combine_gradients.id)]
        schema.final_value_source = ProtocolPath('result', converge_uncertainty.id, enthalpy_of_vaporization.id)

        return schema

    @staticmethod
    def get_default_reweighting_workflow_schema(options):
        """Returns the default workflow to use when estimating this property
        by reweighting existing data.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Set up a protocol to extract both the liquid and gas phase data
        unpack_data_collection = reweighting.UnpackStoredDataCollection('unpack_data_collection_$(data_repl)')
        unpack_data_collection.input_data_path = ReplicatorValue('data_repl')

        # Set up a protocol to extract the liquid phase energy from the existing data.
        extract_liquid_energy = analysis.ExtractAverageStatistic('extract_liquid_energy_$(data_repl)')
        extract_liquid_energy.statistics_type = ObservableType.PotentialEnergy

        liquid_protocols, liquid_data_replicator = generate_base_reweighting_protocols(extract_liquid_energy,
                                                                                       options,
                                                                                       id_suffix='_liquid')

        liquid_protocols.unpack_stored_data.simulation_data_path = ProtocolPath('collection_data_paths[liquid]',
                                                                                unpack_data_collection.id)

        extract_liquid_energy.divisor = ProtocolPath('total_number_of_molecules',
                                                     liquid_protocols.unpack_stored_data.id)
        extract_liquid_energy.statistics_path = ProtocolPath('statistics_file_path',
                                                             liquid_protocols.unpack_stored_data.id)

        # Set up a protocol to extract the gas phase energy from the existing data.
        extract_gas_energy = analysis.ExtractAverageStatistic('extract_gas_energy_$(data_repl)')
        extract_gas_energy.statistics_type = ObservableType.PotentialEnergy

        gas_protocols, gas_data_replicator = generate_base_reweighting_protocols(extract_gas_energy,
                                                                                 options,
                                                                                 id_suffix='_gas')

        # Turn of PBC for the gas phase.
        gas_protocols.reduced_reference_potential.enable_pbc = False
        gas_protocols.reduced_target_potential.enable_pbc = False

        gas_protocols.unpack_stored_data.simulation_data_path = ProtocolPath('collection_data_paths[gas]',
                                                                             unpack_data_collection.id)

        extract_gas_energy.statistics_path = ProtocolPath('statistics_file_path',
                                                          gas_protocols.unpack_stored_data.id)

        # Combine the values to estimate the final enthalpy of vaporization
        energy_of_vaporization = miscellaneous.SubtractValues('energy_of_vaporization')
        energy_of_vaporization.value_b = ProtocolPath('value', gas_protocols.mbar_protocol.id)
        energy_of_vaporization.value_a = ProtocolPath('value', liquid_protocols.mbar_protocol.id)

        ideal_volume = miscellaneous.MultiplyValue('ideal_volume')
        ideal_volume.value = EstimatedQuantity(1.0 * unit.molar_gas_constant,
                                               0.0 * unit.joule / unit.mole / unit.kelvin,
                                               'Universal Constant')
        ideal_volume.multiplier = ProtocolPath('thermodynamic_state.temperature', 'global')

        enthalpy_of_vaporization = miscellaneous.AddValues('enthalpy_of_vaporization')
        enthalpy_of_vaporization.values = [
            ProtocolPath('result', energy_of_vaporization.id),
            ProtocolPath('result', ideal_volume.id)
        ]

        # Combine the data replicators
        data_replicator = ProtocolReplicator(liquid_data_replicator.id)
        data_replicator.protocols_to_replicate = [ProtocolPath('', unpack_data_collection.id),
                                                  *liquid_data_replicator.protocols_to_replicate,
                                                  *gas_data_replicator.protocols_to_replicate]
        data_replicator.template_values = liquid_data_replicator.template_values

        # Set up the liquid phase gradient calculations
        liquid_coordinate_path = ProtocolPath('output_coordinate_path', liquid_protocols.concatenate_trajectories.id)
        liquid_trajectory_path = ProtocolPath('output_trajectory_path', liquid_protocols.concatenate_trajectories.id)

        liquid_gradient_group, liquid_gradient_replicator, liquid_gradient_source = \
            generate_gradient_protocol_group([ProtocolPath('force_field_path', liquid_protocols.unpack_stored_data.id)],
                                             ProtocolPath('force_field_path', 'global'),
                                             liquid_coordinate_path,
                                             liquid_trajectory_path,
                                             'grad',
                                             ProtocolPath('uncorrelated_values', extract_liquid_energy.id),
                                             id_prefix='liquid_',
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   liquid_protocols.mbar_protocol.id))

        # Set up the gas phase gradient calculations
        gas_coordinate_path = ProtocolPath('output_coordinate_path', gas_protocols.concatenate_trajectories.id)
        gas_trajectory_path = ProtocolPath('output_trajectory_path', gas_protocols.concatenate_trajectories.id)

        gas_gradient_group, gas_gradient_replicator, gas_gradient_source = \
            generate_gradient_protocol_group([ProtocolPath('force_field_path', gas_protocols.unpack_stored_data.id)],
                                             ProtocolPath('force_field_path', 'global'),
                                             gas_coordinate_path,
                                             gas_trajectory_path,
                                             'grad',
                                             ProtocolPath('uncorrelated_values', extract_gas_energy.id),
                                             id_prefix='gas_',
                                             enable_pbc=False,
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   gas_protocols.mbar_protocol.id))

        # Combine the gradients.
        combine_gradients = gradients.SubtractGradients('combine_gradients_$(grad)')
        combine_gradients.value_b = gas_gradient_source
        combine_gradients.value_a = liquid_gradient_source

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(liquid_gradient_replicator.id)
        gradient_replicator.protocols_to_replicate = [*liquid_gradient_replicator.protocols_to_replicate,
                                                      *gas_gradient_replicator.protocols_to_replicate,
                                                      ProtocolPath('', combine_gradients.id)]
        gradient_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfVaporization.__name__)
        schema.id = '{}{}'.format(EnthalpyOfVaporization.__name__, 'Schema')

        schema.protocols[unpack_data_collection.id] = unpack_data_collection.schema

        schema.protocols.update({protocol.id: protocol.schema for protocol in liquid_protocols})
        schema.protocols.update({protocol.id: protocol.schema for protocol in gas_protocols})

        schema.protocols[energy_of_vaporization.id] = energy_of_vaporization.schema
        schema.protocols[ideal_volume.id] = ideal_volume.schema
        schema.protocols[enthalpy_of_vaporization.id] = enthalpy_of_vaporization.schema

        schema.protocols[liquid_gradient_group.id] = liquid_gradient_group.schema
        schema.protocols[gas_gradient_group.id] = gas_gradient_group.schema
        schema.protocols[combine_gradients.id] = combine_gradients.schema

        schema.replicators = [data_replicator, gradient_replicator]

        schema.gradients_sources = [ProtocolPath('result', combine_gradients.id)]
        schema.final_value_source = ProtocolPath('result', enthalpy_of_vaporization.id)

        return schema
