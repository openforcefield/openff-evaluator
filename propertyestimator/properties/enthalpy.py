"""
A collection of enthalpy physical property definitions.
"""

from collections import namedtuple

from propertyestimator import unit
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.properties.properties import PhysicalProperty, PropertyPhase, ParameterGradient
from propertyestimator.protocols import analysis, groups, miscellaneous, reweighting, gradients, storage
from propertyestimator.protocols.groups import ProtocolGroup
from propertyestimator.protocols.utils import generate_base_reweighting_protocols, generate_base_simulation_protocols, \
    generate_gradient_protocol_group
from propertyestimator.storage import StoredSimulationData
from propertyestimator.storage.dataclasses import StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow import plugins, WorkflowOptions
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.workflow.schemas import ProtocolReplicator, WorkflowSchema, \
    WorkflowDataCollectionToStore
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


@plugins.register_calculation_protocol()
class BaseWeightByMoleFraction(BaseProtocol):
    """Multiplies a value by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(Substance)
    def component(self, value):
        """The component (e.g water) to which this value belongs."""
        pass

    @protocol_input(Substance)
    def full_substance(self, value):
        """The full substance of which the component of interest is a part."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._value = None
        self._component = None
        self._full_substance = None

        self._weighted_value = None

    def _weight_values(self, mole_fraction):
        """Weights a value by a components mole fraction.

        Parameters
        ----------
        mole_fraction: float
            The mole fraction to weight by.

        Returns
        -------
        Any
            The weighted value.
        """
        raise NotImplementedError()

    def execute(self, directory, available_resources):

        assert len(self._component.components) == 1

        main_component = self._component.components[0]
        amount = self._full_substance.get_amount(main_component)

        if not isinstance(amount, Substance.MoleFraction):

            return PropertyEstimatorException(directory=directory,
                                              message=f'The component {main_component} was given in an '
                                                      f'exact amount, and not a mole fraction')

        self._weighted_value = self._weight_values(amount.value)
        return self._get_output_dictionary()


@plugins.register_calculation_protocol()
class WeightQuantityByMoleFraction(BaseWeightByMoleFraction):
    """Multiplies a quantity by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(EstimatedQuantity)
    def value(self):
        """The value to be weighted."""
        pass

    @protocol_output(EstimatedQuantity)
    def weighted_value(self, value):
        """The value weighted by the `component`s mole fraction as determined from
        the `full_substance`."""
        pass

    def _weight_values(self, mole_fraction):
        """
        Returns
        -------
        EstimatedQuantity
            The weighted value.
        """
        return self._value * mole_fraction


@plugins.register_calculation_protocol()
class WeightGradientByMoleFraction(BaseWeightByMoleFraction):
    """Multiplies a gradient by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(ParameterGradient)
    def value(self):
        """The value to be weighted."""
        pass

    @protocol_output(ParameterGradient)
    def weighted_value(self, value):
        """The value weighted by the `component`s mole fraction as determined from
        the `full_substance`."""
        pass

    def _weight_values(self, mole_fraction):
            """
            Returns
            -------
            ParameterGradient
                The weighted value.
            """
            return ParameterGradient(self._value.key,
                                     self._value.value * mole_fraction)


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
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return EnthalpyOfMixing.get_default_simulation_workflow_schema(options)
        elif calculation_layer == 'ReweightingLayer':
            return EnthalpyOfMixing.get_default_reweighting_workflow_schema(options)

        return None

    @staticmethod
    def _get_enthalpy_protocols(id_suffix, gradient_replicator_id, replicator_id=None,
                                weight_by_mole_fraction=False, substance_reference=None, options=None):

        """Returns the set of protocols which when combined in a workflow
        will yield the enthalpy of a substance.

        Parameters
        ----------
        id_suffix: str
            A suffix to append to the id of each of the returned protocols.
        gradient_replicator_id: str
            The id of the replicator which will clone those protocols which will
            estimate the gradient of the enthalpy with respect to a given parameter.
        replicator_id: str, optional
            The id of the replicator which will be used to clone these protocols.
            This will be appended to the id of each of the returned protocols if
            set.
        weight_by_mole_fraction: bool
            If true, an extra protocol will be added to weight the calculated
            enthalpy by the mole fraction of the component.
        substance_reference: ProtocolPath or PlaceholderInput, optional
            An optional protocol path (or replicator reference) to the substance
            whose enthalpy is being estimated.
        options: WorkflowOptions
            The options to use when setting up the workflows.

        Returns
        -------
        BaseSimulationProtocols
            The protocols used to estimate the enthalpy of a substance.
        ProtocolPath
            A reference to the estimated enthalpy.
        WorkflowSimulationDataToStore
            An object which describes the default data from a simulation to store,
            such as the uncorrelated statistics and configurations.
        ProtocolGroup
            The group of protocols which will calculate the gradient of the enthalpy
            with respect to a given property.
        ProtocolReplicator
            The protocol which will replicate the gradient group for every gradient to
            estimate.
        ProtocolPath
            A reference to the value of the gradient.
        """

        if replicator_id is not None:
            id_suffix = f'{id_suffix}_$({replicator_id})'

        if substance_reference is None:
            substance_reference = ProtocolPath('substance', 'global')

        # Define the protocol which will extract the average enthalpy from
        # the results of a simulation.
        extract_enthalpy = analysis.ExtractAverageStatistic(f'extract_enthalpy{id_suffix}')
        extract_enthalpy.statistics_type = ObservableType.Enthalpy

        # Define the protocols which will run the simulation itself.
        simulation_protocols, value_source, output_to_store = generate_base_simulation_protocols(extract_enthalpy,
                                                                                                 options,
                                                                                                 id_suffix)

        number_of_molecules = ProtocolPath('final_number_of_molecules', simulation_protocols.build_coordinates.id)

        # Divide the enthalpy by the number of molecules in the system
        extract_enthalpy.divisor = number_of_molecules

        # Use the correct substance.
        simulation_protocols.build_coordinates.substance = substance_reference
        simulation_protocols.assign_parameters.substance = substance_reference
        output_to_store.substance = substance_reference

        conditional_group = simulation_protocols.converge_uncertainty

        if weight_by_mole_fraction:

            # The component workflows need an extra step to multiply their enthalpies by their
            # relative mole fraction.
            weight_by_mole_fraction = WeightQuantityByMoleFraction(f'weight_by_mole_fraction{id_suffix}')
            weight_by_mole_fraction.value = ProtocolPath('value', extract_enthalpy.id)
            weight_by_mole_fraction.full_substance = ProtocolPath('substance', 'global')
            weight_by_mole_fraction.component = ReplicatorValue(replicator_id)

            conditional_group.add_protocols(weight_by_mole_fraction)

            value_source = ProtocolPath('weighted_value', conditional_group.id, weight_by_mole_fraction.id)

        # Make sure the weighted value is being used in the conditional comparison.
        if options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks and weight_by_mole_fraction:

            conditional_group.conditions[0].left_hand_value = ProtocolPath('weighted_value.uncertainty',
                                                                           conditional_group.id,
                                                                           weight_by_mole_fraction.id)

        # Set up the gradient calculations
        reweight_enthalpy_template = reweighting.ReweightStatistics('')
        reweight_enthalpy_template.statistics_type = ObservableType.Enthalpy
        reweight_enthalpy_template.statistics_paths = [ProtocolPath('statistics_file_path',
                                                                    conditional_group.id,
                                                                    simulation_protocols.production_simulation.id)]

        coordinate_source = ProtocolPath('output_coordinate_file', simulation_protocols.equilibration_simulation.id)
        trajectory_source = ProtocolPath('trajectory_file_path', simulation_protocols.converge_uncertainty.id,
                                         simulation_protocols.production_simulation.id)
        statistics_source = ProtocolPath('statistics_file_path', simulation_protocols.converge_uncertainty.id,
                                         simulation_protocols.production_simulation.id)

        gradient_group, gradient_replicator, gradient_source = \
            generate_gradient_protocol_group(reweight_enthalpy_template,
                                             ProtocolPath('force_field_path', 'global'),
                                             ProtocolPath('force_field_path', 'global'),
                                             coordinate_source,
                                             trajectory_source,
                                             statistics_source,
                                             replicator_id=gradient_replicator_id,
                                             substance_source=substance_reference,
                                             id_suffix=id_suffix)

        if weight_by_mole_fraction:

            # The component workflows need an extra step to multiply their gradients by their
            # relative mole fraction.
            weight_gradient = WeightGradientByMoleFraction(f'weight_gradient_by_mole_fraction{id_suffix}')
            weight_gradient.value = gradient_source
            weight_gradient.full_substance = ProtocolPath('substance', 'global')
            weight_gradient.component = substance_reference

            gradient_group.add_protocols(weight_gradient)
            gradient_source = ProtocolPath('weighted_value', gradient_group.id, weight_gradient.id)

        scale_gradient = gradients.DivideGradientByScalar(f'scale_gradient{id_suffix}')
        scale_gradient.value = gradient_source
        scale_gradient.divisor = number_of_molecules

        gradient_group.add_protocols(scale_gradient)
        gradient_source = ProtocolPath('result', gradient_group.id, scale_gradient.id)

        return (simulation_protocols, value_source, output_to_store,
                gradient_group, gradient_replicator, gradient_source)

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

        # Define the id of the replicator which will clone the gradient protocols
        # for each gradient key to be estimated.
        gradient_replicator_id = 'gradient_replicator'

        # Set up a general workflow for calculating the enthalpy of one of the system components.
        # Here we affix a prefix which contains the special string $(comp_index). Protocols which are
        # replicated by a replicator will have the $(comp_index) tag in their id replaced by the index
        # of the replication.
        component_replicator_id = 'component_replicator'
        component_substance = ReplicatorValue(component_replicator_id)

        (component_protocols,
         component_enthalpies,
         component_output,
         component_gradient_group,
         component_gradient_replicator,
         component_gradient) = EnthalpyOfMixing._get_enthalpy_protocols(id_suffix='_component',
                                                                        gradient_replicator_id=gradient_replicator_id,
                                                                        replicator_id=component_replicator_id,
                                                                        weight_by_mole_fraction=True,
                                                                        substance_reference=component_substance,
                                                                        options=options)

        # Set up a workflow to calculate the enthalpy of the full, mixed system.
        (full_system_protocols,
         full_system_enthalpy,
         full_output,
         full_system_gradient_group,
         full_system_gradient_replicator,
         full_system_gradient) = EnthalpyOfMixing._get_enthalpy_protocols(id_suffix='_full',
                                                                          gradient_replicator_id=gradient_replicator_id,
                                                                          options=options)

        # Finally, set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the mixed system enthalpy.
        add_component_enthalpies = miscellaneous.AddValues('add_component_enthalpies')
        add_component_enthalpies.values = component_enthalpies

        calculate_enthalpy_of_mixing = miscellaneous.SubtractValues('calculate_enthalpy_of_mixing')
        calculate_enthalpy_of_mixing.value_b = full_system_enthalpy
        calculate_enthalpy_of_mixing.value_a = ProtocolPath('result', add_component_enthalpies.id)

        # Create the replicator object which defines how the pure component
        # enthalpy estimation protocols will be replicated for each component.
        component_replicator = ProtocolReplicator(replicator_id=component_replicator_id)
        component_replicator.template_values = ProtocolPath('components', 'global')

        # Combine the gradients.
        add_component_gradients = gradients.AddGradients(f'add_component_gradients'
                                                         f'_$({gradient_replicator_id})'
                                                         f'_$({component_replicator_id})')
        add_component_gradients.values = [component_gradient]

        combine_gradients = gradients.SubtractGradients(f'combine_gradients_$({gradient_replicator_id})')
        combine_gradients.value_b = full_system_gradient
        combine_gradients.value_a = ProtocolPath('result', add_component_gradients.id)

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(replicator_id=gradient_replicator_id)
        gradient_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

        # Build the final workflow schema
        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        schema.protocols = {
            component_protocols.build_coordinates.id: component_protocols.build_coordinates.schema,
            component_protocols.assign_parameters.id: component_protocols.assign_parameters.schema,
            component_protocols.energy_minimisation.id: component_protocols.energy_minimisation.schema,
            component_protocols.equilibration_simulation.id: component_protocols.equilibration_simulation.schema,
            component_protocols.converge_uncertainty.id: component_protocols.converge_uncertainty.schema,

            full_system_protocols.build_coordinates.id: full_system_protocols.build_coordinates.schema,
            full_system_protocols.assign_parameters.id: full_system_protocols.assign_parameters.schema,
            full_system_protocols.energy_minimisation.id: full_system_protocols.energy_minimisation.schema,
            full_system_protocols.equilibration_simulation.id: full_system_protocols.equilibration_simulation.schema,
            full_system_protocols.converge_uncertainty.id: full_system_protocols.converge_uncertainty.schema,

            component_protocols.extract_uncorrelated_trajectory.id:
                component_protocols.extract_uncorrelated_trajectory.schema,
            component_protocols.extract_uncorrelated_statistics.id:
                component_protocols.extract_uncorrelated_statistics.schema,

            full_system_protocols.extract_uncorrelated_trajectory.id:
                full_system_protocols.extract_uncorrelated_trajectory.schema,
            full_system_protocols.extract_uncorrelated_statistics.id:
                full_system_protocols.extract_uncorrelated_statistics.schema,

            add_component_enthalpies.id: add_component_enthalpies.schema,
            calculate_enthalpy_of_mixing.id: calculate_enthalpy_of_mixing.schema,
            
            component_gradient_group.id: component_gradient_group.schema,
            full_system_gradient_group.id: full_system_gradient_group.schema,
            add_component_gradients.id: add_component_gradients.schema,
            combine_gradients.id: combine_gradients.schema
        }

        schema.replicators = [gradient_replicator, component_replicator]

        # Finally, tell the schemas where to look for its final values.
        schema.gradients_sources = [ProtocolPath('result', combine_gradients.id)]
        schema.final_value_source = ProtocolPath('result', calculate_enthalpy_of_mixing.id)

        schema.outputs_to_store = {
            'full_system': full_output,
            f'component_$({component_replicator_id})': component_output
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

        component_replicator_id = 'component_replicator'
        component_data_replicator_id = f'component_$({component_replicator_id})_replicator'
        
        full_data_replicator_id = 'full_data_replicator'

        # Set up the protocols which will reweight data for the full system.
        extract_full_enthalpy = analysis.ExtractAverageStatistic(f'extract_enthalpy_'
                                                                 f'$({full_data_replicator_id})_full')
        reweight_full_enthalpy = reweighting.ReweightStatistics('reweight_full_enthalpy')

        extract_full_enthalpy.statistics_type = ObservableType.Enthalpy
        reweight_full_enthalpy.statistics_type = ObservableType.Enthalpy

        (full_system_protocols,
         full_system_data_replicator) = generate_base_reweighting_protocols(analysis_protocol=extract_full_enthalpy,
                                                                            mbar_protocol=reweight_full_enthalpy,
                                                                            workflow_options=options,
                                                                            replicator_id=full_data_replicator_id,
                                                                            id_suffix='_full')

        extract_full_enthalpy.statistics_path = ProtocolPath('statistics_file_path',
                                                              full_system_protocols.unpack_stored_data.id)

        # Set up the protocols which will reweight data for each of the components.
        extract_component_enthalpy = analysis.ExtractAverageStatistic(f'extract_enthalpy_'
                                                                      f'$({component_data_replicator_id})_component_'
                                                                      f'$({component_replicator_id})')

        reweight_component_enthalpy = reweighting.ReweightStatistics(f'reweight_enthalpy_component_'
                                                                     f'$({component_replicator_id})')

        extract_component_enthalpy.statistics_type = ObservableType.Enthalpy
        reweight_component_enthalpy.statistics_type = ObservableType.Enthalpy

        (component_protocols,
         component_data_replicator) = generate_base_reweighting_protocols(analysis_protocol=extract_component_enthalpy,
                                                                          mbar_protocol=reweight_component_enthalpy,
                                                                          workflow_options=options,
                                                                          replicator_id=component_data_replicator_id,
                                                                          id_suffix=f'_component_'
                                                                                    f'$({component_replicator_id})')

        extract_component_enthalpy.statistics_path = ProtocolPath('statistics_file_path',
                                                                  component_protocols.unpack_stored_data.id)

        # Make sure the replicator is only replicating over data from the component component.
        component_data_replicator.template_values = ProtocolPath(f'component_data[$({component_replicator_id})]',
                                                                 'global')

        # Set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the full system enthalpy.
        weight_by_mole_fraction = WeightQuantityByMoleFraction(f'weight_comp_$({component_replicator_id})')
        weight_by_mole_fraction.value = ProtocolPath('value', component_protocols.mbar_protocol.id)
        weight_by_mole_fraction.full_substance = ProtocolPath('substance', 'global')
        weight_by_mole_fraction.component = ReplicatorValue(component_replicator_id)

        # Divide by the component enthalpies by the number of molecules in the system
        # TODO cleanup replicators so can target a single replicated protocol rather
        #      than a list if possible
        divisor_data = storage.UnpackStoredSimulationData('divisor_data')
        divisor_data.simulation_data_path = ProtocolPath('full_system_data[0]', 'global')

        component_divide_by_molecules = miscellaneous.DivideValue(f'divide_by_molecules_$({component_replicator_id})')
        component_divide_by_molecules.value = ProtocolPath('weighted_value', weight_by_mole_fraction.id)
        component_divide_by_molecules.divisor = ProtocolPath('total_number_of_molecules', divisor_data.id)

        # Divide by the full_system enthalpy by the number of molecules in the system
        full_system_divide_by_molecules = miscellaneous.DivideValue('divide_by_full_system_molecules')
        full_system_divide_by_molecules.value = ProtocolPath('value', full_system_protocols.mbar_protocol.id)
        full_system_divide_by_molecules.divisor = ProtocolPath('total_number_of_molecules', divisor_data.id)

        add_component_enthalpies = miscellaneous.AddValues('add_component_enthalpies')
        add_component_enthalpies.values = [ProtocolPath('result', component_divide_by_molecules.id)]

        calculate_enthalpy_of_mixing = miscellaneous.SubtractValues('calculate_enthalpy_of_mixing')
        calculate_enthalpy_of_mixing.value_b = ProtocolPath('result', full_system_divide_by_molecules.id)
        calculate_enthalpy_of_mixing.value_a = ProtocolPath('result', add_component_enthalpies.id)

        # Set up a replicator that will re-run the component reweighting workflow for each
        # component in the system.
        component_replicator = ProtocolReplicator(replicator_id=component_replicator_id)
        component_replicator.template_values = ProtocolPath('components', 'global')

        # Build the final workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        schema.protocols = dict()

        schema.protocols[divisor_data.id] = divisor_data.schema

        schema.protocols.update({protocol.id: protocol.schema for protocol in full_system_protocols})
        schema.protocols.update({protocol.id: protocol.schema for protocol in component_protocols})

        schema.protocols[weight_by_mole_fraction.id] = weight_by_mole_fraction.schema
        schema.protocols[component_divide_by_molecules.id] = component_divide_by_molecules.schema
        schema.protocols[full_system_divide_by_molecules.id] = full_system_divide_by_molecules.schema
        schema.protocols[add_component_enthalpies.id] = add_component_enthalpies.schema
        schema.protocols[calculate_enthalpy_of_mixing.id] = calculate_enthalpy_of_mixing.schema

        schema.replicators = [
            full_system_data_replicator,
            component_replicator,
            component_data_replicator
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
        elif calculation_layer == 'ReweightingLayer':
            return EnthalpyOfVaporization.get_default_reweighting_workflow_schema(options)

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
        reweight_potential_template = reweighting.ReweightStatistics('')
        reweight_potential_template.statistics_type = ObservableType.PotentialEnergy

        liquid_coordinate_source = ProtocolPath('output_coordinate_file', liquid_protocols.equilibration_simulation.id)
        liquid_trajectory_source = ProtocolPath('trajectory_file_path', converge_uncertainty.id,
                                                liquid_protocols.production_simulation.id)
        liquid_statistics_source = ProtocolPath('statistics_file_path', liquid_protocols.converge_uncertainty.id,
                                                liquid_protocols.production_simulation.id)

        liquid_gradient_group, liquid_gradient_replicator, liquid_gradient_source = \
            generate_gradient_protocol_group(reweight_potential_template,
                                             [ProtocolPath('force_field_path', 'global')],
                                             ProtocolPath('force_field_path', 'global'),
                                             liquid_coordinate_source,
                                             liquid_trajectory_source,
                                             liquid_statistics_source,
                                             id_suffix='_liquid')

        # Set up the gas gradient calculations
        gas_coordinate_source = ProtocolPath('output_coordinate_file', gas_protocols.equilibration_simulation.id)
        gas_trajectory_source = ProtocolPath('trajectory_file_path', converge_uncertainty.id,
                                                gas_protocols.production_simulation.id)
        gas_statistics_source = ProtocolPath('statistics_file_path', gas_protocols.converge_uncertainty.id,
                                             gas_protocols.production_simulation.id)

        gas_gradient_group, gas_gradient_replicator, gas_gradient_source = \
            generate_gradient_protocol_group(reweight_potential_template,
                                             [ProtocolPath('force_field_path', 'global')],
                                             ProtocolPath('force_field_path', 'global'),
                                             gas_coordinate_source,
                                             gas_trajectory_source,
                                             gas_statistics_source,
                                             id_suffix='_gas',
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
        unpack_data_collection = storage.UnpackStoredDataCollection('unpack_data_collection_$(data_repl)')
        unpack_data_collection.input_data_path = ReplicatorValue('data_repl')

        # Set up a protocol to extract the liquid phase energy from the existing data.
        extract_liquid_energy = analysis.ExtractAverageStatistic('extract_liquid_energy_$(data_repl)')
        extract_liquid_energy.statistics_type = ObservableType.PotentialEnergy
        reweight_liquid_energy = reweighting.ReweightStatistics('reweight_liquid_energy')
        reweight_liquid_energy.statistics_type = ObservableType.PotentialEnergy

        liquid_protocols, liquid_data_replicator = generate_base_reweighting_protocols(extract_liquid_energy,
                                                                                       reweight_liquid_energy,
                                                                                       options,
                                                                                       id_suffix='_liquid')

        liquid_protocols.unpack_stored_data.simulation_data_path = ProtocolPath('collection_data_paths[liquid]',
                                                                                unpack_data_collection.id)

        extract_liquid_energy.divisor = ProtocolPath('total_number_of_molecules',
                                                     liquid_protocols.unpack_stored_data.id)

        # Set up a protocol to extract the gas phase energy from the existing data.
        extract_gas_energy = analysis.ExtractAverageStatistic('extract_gas_energy_$(data_repl)')
        extract_gas_energy.statistics_type = ObservableType.PotentialEnergy
        reweight_gas_energy = reweighting.ReweightStatistics('reweight_gas_energy')
        reweight_gas_energy.statistics_type = ObservableType.PotentialEnergy

        gas_protocols, gas_data_replicator = generate_base_reweighting_protocols(extract_gas_energy,
                                                                                 reweight_gas_energy,
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
        data_replicator.template_values = liquid_data_replicator.template_values

        # Set up the liquid phase gradient calculations
        reweight_potential_template = reweighting.ReweightStatistics('')
        reweight_potential_template.statistics_type = ObservableType.PotentialEnergy

        liquid_coordinate_path = ProtocolPath('output_coordinate_path', liquid_protocols.concatenate_trajectories.id)
        liquid_trajectory_path = ProtocolPath('output_trajectory_path', liquid_protocols.concatenate_trajectories.id)

        liquid_gradient_group, liquid_gradient_replicator, liquid_gradient_source = \
            generate_gradient_protocol_group(reweight_potential_template,
                                             ProtocolPath('force_field_path', liquid_protocols.unpack_stored_data.id),
                                             ProtocolPath('force_field_path', 'global'),
                                             liquid_coordinate_path,
                                             liquid_trajectory_path,
                                             replicator_id='grad',
                                             id_suffix='_liquid',
                                             use_subset_of_force_field=False,
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   liquid_protocols.mbar_protocol.id))

        # Set up the gas phase gradient calculations
        gas_coordinate_path = ProtocolPath('output_coordinate_path', gas_protocols.concatenate_trajectories.id)
        gas_trajectory_path = ProtocolPath('output_trajectory_path', gas_protocols.concatenate_trajectories.id)

        gas_gradient_group, gas_gradient_replicator, gas_gradient_source = \
            generate_gradient_protocol_group(reweight_potential_template,
                                             ProtocolPath('force_field_path', gas_protocols.unpack_stored_data.id),
                                             ProtocolPath('force_field_path', 'global'),
                                             gas_coordinate_path,
                                             gas_trajectory_path,
                                             replicator_id='grad',
                                             id_suffix='_gas',
                                             use_subset_of_force_field=False,
                                             enable_pbc=False,
                                             effective_sample_indices=ProtocolPath('effective_sample_indices',
                                                                                   gas_protocols.mbar_protocol.id))

        # Combine the gradients.
        # TODO - this is an obvious failing of the current workflow code that we have to
        #        resort to this to get the number of molecules. This should be fixed ASAP
        #        once a better typing system is in.
        divide_data_collection = storage.UnpackStoredDataCollection('divide_data_collection')
        divide_data_collection.input_data_path = ProtocolPath('full_system_data[0]', 'global')

        divide_data = storage.UnpackStoredSimulationData('molecule_count')
        divide_data.simulation_data_path = ProtocolPath('collection_data_paths[liquid]',
                                                        divide_data_collection.id)

        scale_liquid_gradient = gradients.DivideGradientByScalar('scale_liquid_gradient_$(grad)')
        scale_liquid_gradient.value = liquid_gradient_source
        scale_liquid_gradient.divisor = ProtocolPath('total_number_of_molecules', divide_data.id)

        combine_gradients = gradients.SubtractGradients('combine_gradients_$(grad)')
        combine_gradients.value_b = gas_gradient_source
        combine_gradients.value_a = ProtocolPath('result', scale_liquid_gradient.id)

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(liquid_gradient_replicator.id)
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
        schema.protocols[divide_data_collection.id] = divide_data_collection.schema
        schema.protocols[divide_data.id] = divide_data.schema
        schema.protocols[scale_liquid_gradient.id] = scale_liquid_gradient.schema
        schema.protocols[combine_gradients.id] = combine_gradients.schema

        schema.replicators = [data_replicator, gradient_replicator]

        schema.gradients_sources = [ProtocolPath('result', combine_gradients.id)]
        schema.final_value_source = ProtocolPath('result', enthalpy_of_vaporization.id)

        return schema
