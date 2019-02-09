"""
A collection of enthalpy physical property definitions.
"""

from collections import namedtuple

from simtk import unit

from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.properties.properties import PhysicalProperty
from propertyestimator.substances import Mixture
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.utils.serialization import PolymorphicDataType
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow import WorkflowSchema, plugins, groups
from propertyestimator.workflow import protocols
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, PlaceholderInput


@plugins.register_calculation_protocol()
class WeightValueByMoleFraction(protocols.BaseProtocol):
    """Multiplies a value by the mole fraction of a component
    in a mixture substance.
    """
    @protocol_input(unit.Quantity)
    def value(self, value):
        """The system object which defines the forces present in the system."""
        pass

    @protocol_input(Mixture)
    def component(self, value):
        """The component (e.g water) to which this value belongs."""
        pass

    @protocol_input(Mixture)
    def full_substance(self, value):
        """The full substance of which the component of interest is a part."""
        pass

    @protocol_output(unit.Quantity)
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
        mole_fraction = 0.0

        for component in self._full_substance.components:

            if component.smiles != main_component.smiles:
                continue

            mole_fraction = component.mole_fraction

        self._weighted_value = mole_fraction * self.value

        if not isinstance(self._weighted_value, unit.Quantity):
            self._weighted_value = unit.Quantity(self._weighted_value, None)

        return self._get_output_dictionary()


@register_estimable_property()
@register_thermoml_property(thermoml_string='Excess molar enthalpy (molar enthalpy of mixing), kJ/mol')
class EnthalpyOfMixing(PhysicalProperty):
    """A class representation of an enthalpy of mixing property"""

    EnthalpyWorkflow = namedtuple('EnthalpySchema', 'build_coordinates '
                                                    'assign_topology '
                                                    'energy_minimisation '
                                                    'npt_equilibration '
                                                    'npt_production '
                                                    'extract_enthalpy')

    @staticmethod
    def get_enthalpy_workflow(id_prefix=''):
        """Returns the set of protocols which when combined in a workflow
        will yield the enthalpy of a substance.

        Parameters
        ----------
        id_prefix: str
            A prefix to append to the id of each of the returned protocols.

        Returns
        -------
        EnthalpyOfMixing.EnthalpyWorkflow
            The protocols used to estimate the enthalpy of a substance.
        """

        build_coordinates = protocols.BuildCoordinatesPackmol(id_prefix + 'build_coordinates')

        build_coordinates.substance = ProtocolPath('substance', 'global')

        assign_topology = protocols.BuildSmirnoffTopology(id_prefix + 'build_topology')

        assign_topology.force_field_path = ProtocolPath('force_field_path', 'global')

        assign_topology.coordinate_file_path = ProtocolPath('coordinate_file_path', build_coordinates.id)
        assign_topology.substance = ProtocolPath('substance', 'global')

        # Equilibration
        energy_minimisation = protocols.RunEnergyMinimisation(id_prefix + 'energy_minimisation')

        energy_minimisation.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
        energy_minimisation.system = ProtocolPath('system', assign_topology.id)

        npt_equilibration = protocols.RunOpenMMSimulation(id_prefix + 'npt_equilibration')

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps = 2  # Debug settings.
        npt_equilibration.output_frequency = 1  # Debug settings.

        npt_equilibration.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system = ProtocolPath('system', assign_topology.id)

        # Production
        npt_production = protocols.RunOpenMMSimulation(id_prefix + 'npt_production')

        npt_production.ensemble = Ensemble.NPT

        npt_production.steps = 2  # Debug settings.
        npt_production.output_frequency = 1  # Debug settings.

        npt_production.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system = ProtocolPath('system', assign_topology.id)

        # Analysis
        extract_enthalpy = protocols.ExtractAverageStatistic(id_prefix + 'extract_enthalpy')

        extract_enthalpy.statistics_type = ObservableType.Enthalpy
        extract_enthalpy.statistics_path = ProtocolPath('statistics_file_path', npt_production.id)

        return EnthalpyOfMixing.EnthalpyWorkflow(build_coordinates, assign_topology,
                                                 energy_minimisation, npt_equilibration,
                                                 npt_production, extract_enthalpy)

    @staticmethod
    def get_default_calculation_schema():

        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = '{}{}'.format(EnthalpyOfMixing.__name__, 'Schema')

        # Set up a general workflow for calculating the enthalpy of one of the system components.
        # Here we affix a prefix which contains the special string $index. Protocols which are
        # replicated by a replicator will have the $index tag in their id replaced by the index
        # of the replication.
        component_workflow = EnthalpyOfMixing.get_enthalpy_workflow('component_$index_')

        # Set the substance of the build_coordinates and assign_topology protocols
        # as a placeholder for now - these will be later set by the replicator.
        component_workflow.build_coordinates.substance = PlaceholderInput()
        component_workflow.assign_topology.substance = PlaceholderInput()

        # The component workflows need an extra step to multiply their enthalpies by their
        # relative mole fraction.
        weight_by_mole_fraction = WeightValueByMoleFraction('component_$index_weight_by_mole_fraction')

        weight_by_mole_fraction.value = ProtocolPath('value', component_workflow.extract_enthalpy.id)
        weight_by_mole_fraction.full_substance = ProtocolPath('substance', 'global')

        # Again, set the component as a placeholder which will be set by the replicator.
        weight_by_mole_fraction.component = PlaceholderInput()

        # Set up a workflow to calculate the enthalpy of the full, mixed system.
        mixed_system_workflow = EnthalpyOfMixing.get_enthalpy_workflow('mixed_')

        # Finally, set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the mixed system enthalpy.
        add_component_enthalpies = protocols.AddQuantities('add_component_enthalpies')

        # Although we only give a list of a single ProtocolPath pointing to our template
        # component workflow's `weight_by_mole_fraction` protocol, the replicator
        # will actually populate this list with references to all of the newly generated
        # protocols of the individual components.
        add_component_enthalpies.values = [ProtocolPath('weighted_value', weight_by_mole_fraction.id)]

        calculate_enthalpy_of_mixing = protocols.SubtractQuantities('calculate_enthalpy_of_mixing')

        calculate_enthalpy_of_mixing.value_b = ProtocolPath('value', mixed_system_workflow.extract_enthalpy.id)
        calculate_enthalpy_of_mixing.value_a = ProtocolPath('result', add_component_enthalpies.id)

        # Set up converge uncertainty
        converge_uncertainty = groups.ConditionalGroup('converge_uncertainty')

        converge_uncertainty.add_protocols(component_workflow.npt_production,
                                           component_workflow.extract_enthalpy,
                                           weight_by_mole_fraction,
                                           mixed_system_workflow.npt_production,
                                           mixed_system_workflow.extract_enthalpy,
                                           add_component_enthalpies,
                                           calculate_enthalpy_of_mixing)

        condition = groups.ConditionalGroup.Condition()

        condition.left_hand_value = ProtocolPath('result',
                                                 converge_uncertainty.id,
                                                 calculate_enthalpy_of_mixing.id)

        condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

        condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

        converge_uncertainty.add_condition(condition)
        converge_uncertainty.max_iterations = 1

        schema.protocols[converge_uncertainty.id] = converge_uncertainty.schema

        for component_protocol in component_workflow:

            if component_protocol.id not in converge_uncertainty.protocols:
                schema.protocols[component_protocol.id] = component_protocol.schema

        for mixed_protocol in mixed_system_workflow:

            if mixed_protocol.id not in converge_uncertainty.protocols:
                schema.protocols[mixed_protocol.id] = mixed_protocol.schema

        # Create the replicator object which defines how the pure component
        # enthalpy estimation workflow will be replicated for each component.
        component_replicator = ProtocolReplicator()

        protocols_to_replicate = []

        # Pass it paths to the protocols to be replicated.
        for component_protocol in component_workflow:

            path_to_protocol = ProtocolPath('', component_protocol.id)

            if component_protocol.id in converge_uncertainty.protocols:
                path_to_protocol.prepend_protocol_id(converge_uncertainty.id)

            protocols_to_replicate.append(path_to_protocol)

        protocols_to_replicate.append(ProtocolPath('', converge_uncertainty.id, weight_by_mole_fraction.id))

        component_replicator.protocols_to_replicate = protocols_to_replicate

        # We now tell the replicator that when it replicates the component workflows,
        # it should replace the substance place holder values with ones specified in the
        # replicators `template_values` list.
        component_replicator.template_targets = [
            ProtocolPath('substance', component_workflow.build_coordinates.id),
            ProtocolPath('substance', component_workflow.assign_topology.id),
            ProtocolPath('component', converge_uncertainty.id, weight_by_mole_fraction.id)
        ]

        # Tell the replicator to take the components of a properties substance,
        # and pass these to the replicated workflows being produced, and in particular,
        # the inputs specified by the `template_targets`
        component_replicator.template_values = PolymorphicDataType(ProtocolPath('components', 'global'))

        schema.replicators = [component_replicator]

        # Finally, tell the schemas where to look for its final values.
        schema.final_value_source = ProtocolPath('result', converge_uncertainty.id,
                                                           calculate_enthalpy_of_mixing.id)

        schema.final_uncertainty_source = ProtocolPath('result', converge_uncertainty.id,
                                                                 calculate_enthalpy_of_mixing.id)

        # TODO: Replace with the real values.
        schema.final_coordinate_source = ProtocolPath('coordinate_file_path',
                                                      mixed_system_workflow.build_coordinates.id)

        schema.final_trajectory_source = ProtocolPath('trajectory_file_path', converge_uncertainty.id,
                                                      mixed_system_workflow.npt_production.id)

        return schema
