import abc
from typing import Dict, Optional, Tuple

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED, PlaceholderValue
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.layers.reweighting import ReweightingSchema
from openff.evaluator.layers.simulation import SimulationSchema
from openff.evaluator.protocols import analysis, miscellaneous
from openff.evaluator.protocols.utils import (
    generate_reweighting_protocols,
    generate_simulation_protocols,
)
from openff.evaluator.storage.query import SimulationDataQuery, SubstanceQuery
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow.schemas import ProtocolReplicator, WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue


class EstimableExcessProperty(PhysicalProperty, abc.ABC):
    """A base class for estimable excess physical properties, such as enthalpies of
    mixing or excess molar volumes, which provides a common framework for defining
    estimation workflows.
    """

    @classmethod
    @abc.abstractmethod
    def _observable_type(cls) -> ObservableType:
        """The type of excess observable that this class corresponds to."""
        raise NotImplementedError

    @classmethod
    def _n_molecules_divisor(
        cls, n_molecules: ProtocolPath, suffix: Optional[str] = None
    ) -> Tuple[ProtocolPath, Optional[miscellaneous.DivideValue]]:
        """Returns the number of molecules to scale the value of the observable by.
        For energies this is just the total number of molecules in the box as they are
        already in units per mole. For other observables this is the total number of
        molecules divided by the Avogadro constant.

        Parameters
        ----------
        n_molecules
            A reference to the number of molecules in the simulation box.
        suffix
            An optional string to append to the id of the protocol which will
            normalize the number of molecules by the Avogadro constant. This
            argument is only used for observables which aren't energies.

        Returns
        -------
            A reference to the divisor as well as optionally the protocol from
            which it is computed.
        """

        suffix = "" if suffix is None else suffix

        n_molar_molecules = None

        if cls._observable_type() in [
            ObservableType.Temperature,
            ObservableType.Volume,
            ObservableType.Density,
        ]:

            n_molar_molecules = miscellaneous.DivideValue(f"n_molar_molecules{suffix}")
            n_molar_molecules.value = n_molecules
            n_molar_molecules.divisor = (1.0 * unit.avogadro_constant).to("mole**-1")

            n_molecules = ProtocolPath("result", n_molar_molecules.id)

        return n_molecules, n_molar_molecules

    @classmethod
    def default_simulation_schema(
        cls,
        absolute_tolerance=UNDEFINED,
        relative_tolerance=UNDEFINED,
        n_molecules=1000,
    ) -> SimulationSchema:
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Parameters
        ----------
        absolute_tolerance: pint.Quantity, optional
            The absolute tolerance to estimate the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = SimulationSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        use_target_uncertainty = (
            absolute_tolerance != UNDEFINED or relative_tolerance != UNDEFINED
        )

        # Define the protocols to use for the fully mixed system.
        (
            mixture_protocols,
            mixture_value,
            mixture_stored_data,
        ) = generate_simulation_protocols(
            analysis.AverageObservable("extract_observable_mixture"),
            use_target_uncertainty,
            id_suffix="_mixture",
            n_molecules=n_molecules,
        )
        # Specify the average observable which should be estimated.
        mixture_protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{cls._observable_type().value}]",
            mixture_protocols.production_simulation.id,
        )
        (
            mixture_protocols.analysis_protocol.divisor,
            mixture_n_molar_molecules,
        ) = cls._n_molecules_divisor(
            ProtocolPath(
                "output_number_of_molecules", mixture_protocols.build_coordinates.id
            ),
            "_mixture",
        )

        # Define the protocols to use for each component, creating a replicator that
        # will copy these for each component in the mixture substance.
        component_replicator = ProtocolReplicator("component_replicator")
        component_replicator.template_values = ProtocolPath("components", "global")
        component_substance = ReplicatorValue(component_replicator.id)

        component_protocols, _, component_stored_data = generate_simulation_protocols(
            analysis.AverageObservable("extract_observable_component"),
            use_target_uncertainty,
            id_suffix=f"_component_{component_replicator.placeholder_id}",
            n_molecules=n_molecules,
        )
        # Make sure the protocols point to the correct substance.
        component_protocols.build_coordinates.substance = component_substance
        # Specify the average observable which should be estimated.
        component_protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{cls._observable_type().value}]",
            component_protocols.production_simulation.id,
        )
        (
            component_protocols.analysis_protocol.divisor,
            component_n_molar_molecules,
        ) = cls._n_molecules_divisor(
            ProtocolPath(
                "output_number_of_molecules", component_protocols.build_coordinates.id
            ),
            f"_component_{component_replicator.placeholder_id}",
        )

        # Weight the component value by the mole fraction.
        weight_by_mole_fraction = miscellaneous.WeightByMoleFraction(
            f"weight_by_mole_fraction_{component_replicator.placeholder_id}"
        )
        weight_by_mole_fraction.value = ProtocolPath(
            "value", component_protocols.analysis_protocol.id
        )
        weight_by_mole_fraction.full_substance = ProtocolPath("substance", "global")
        weight_by_mole_fraction.component = component_substance

        component_protocols.converge_uncertainty.add_protocols(weight_by_mole_fraction)

        # Make sure the convergence criteria is set to use the per component
        # uncertainty target.
        if use_target_uncertainty:
            component_protocols.converge_uncertainty.conditions[
                0
            ].right_hand_value = ProtocolPath("per_component_uncertainty", "global")

        # Finally, set up the protocols which will be responsible for adding together
        # the component observables, and subtracting these from the mixture system value.
        add_component_observables = miscellaneous.AddValues("add_component_observables")
        add_component_observables.values = ProtocolPath(
            "weighted_value",
            component_protocols.converge_uncertainty.id,
            weight_by_mole_fraction.id,
        )

        calculate_excess_observable = miscellaneous.SubtractValues(
            "calculate_excess_observable"
        )
        calculate_excess_observable.value_b = mixture_value
        calculate_excess_observable.value_a = ProtocolPath(
            "result", add_component_observables.id
        )

        # Build the final workflow schema
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            component_protocols.build_coordinates.schema,
            component_protocols.assign_parameters.schema,
            component_protocols.energy_minimisation.schema,
            component_protocols.equilibration_simulation.schema,
            component_protocols.converge_uncertainty.schema,
            component_protocols.decorrelate_trajectory.schema,
            component_protocols.decorrelate_observables.schema,
            mixture_protocols.build_coordinates.schema,
            mixture_protocols.assign_parameters.schema,
            mixture_protocols.energy_minimisation.schema,
            mixture_protocols.equilibration_simulation.schema,
            mixture_protocols.converge_uncertainty.schema,
            mixture_protocols.decorrelate_trajectory.schema,
            mixture_protocols.decorrelate_observables.schema,
            add_component_observables.schema,
            calculate_excess_observable.schema,
        ]

        if component_n_molar_molecules is not None:
            schema.protocol_schemas.append(component_n_molar_molecules.schema)
        if mixture_n_molar_molecules is not None:
            schema.protocol_schemas.append(mixture_n_molar_molecules.schema)

        schema.protocol_replicators = [component_replicator]

        schema.final_value_source = ProtocolPath(
            "result", calculate_excess_observable.id
        )

        schema.outputs_to_store = {
            "full_system": mixture_stored_data,
            f"component_{component_replicator.placeholder_id}": component_stored_data,
        }

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @classmethod
    def _default_reweighting_storage_query(cls) -> Dict[str, SimulationDataQuery]:
        """Returns the default storage queries to use when retrieving cached simulation
        data to reweight.

        This will include one query (with the key `"full_system_data"`) to return data
        for the full mixture system, and another query (with the key `"component_data"`)
        which will include data for each pure component in the system.

        Returns
        -------
            The dictionary of queries.
        """
        mixture_data_query = SimulationDataQuery()
        mixture_data_query.substance = PlaceholderValue()
        mixture_data_query.property_phase = PropertyPhase.Liquid

        # Set up a query which will return the data of each
        # individual component in the system.
        component_query = SubstanceQuery()
        component_query.components_only = True

        component_data_query = SimulationDataQuery()
        component_data_query.property_phase = PropertyPhase.Liquid
        component_data_query.substance = PlaceholderValue()
        component_data_query.substance_query = component_query

        return {
            "full_system_data": mixture_data_query,
            "component_data": component_data_query,
        }

    @classmethod
    def _default_reweighting_schema(
        cls,
        observable_type: ObservableType,
        absolute_tolerance: unit.Quantity = UNDEFINED,
        relative_tolerance: float = UNDEFINED,
        n_effective_samples: int = 50,
    ) -> ReweightingSchema:
        """Returns the default calculation schema to use when estimating this class of
        property by re-weighting cached simulation data.

        This internal implementation allows re-weighting a different observable than
        may be specified by the `_observable_type` class property.

        Parameters
        ----------
        absolute_tolerance
            The absolute tolerance to estimate the property to within.
        relative_tolerance
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_effective_samples
            The minimum number of effective samples to require when
            reweighting the cached simulation data.

        Returns
        -------
            The default re-weighting calculation schema.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = ReweightingSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        # Set up the storage queries
        calculation_schema.storage_queries = cls._default_reweighting_storage_query()

        # Define the protocols which will re-weight the observable computed for the
        # fully mixed system.
        mixture_protocols, mixture_data_replicator = generate_reweighting_protocols(
            observable_type,
            "mixture_data_replicator",
            "_mixture",
        )
        mixture_protocols.reweight_observable.required_effective_samples = (
            n_effective_samples
        )

        divide_by_mixture_molecules = miscellaneous.DivideValue(
            "divide_by_mixture_molecules"
        )
        divide_by_mixture_molecules.value = ProtocolPath(
            "value", mixture_protocols.reweight_observable.id
        )
        (
            divide_by_mixture_molecules.divisor,
            mixture_n_molar_molecules,
        ) = cls._n_molecules_divisor(
            ProtocolPath(
                "total_number_of_molecules",
                mixture_protocols.unpack_stored_data.id.replace(
                    mixture_data_replicator.placeholder_id, "0"
                ),
            ),
            "_mixture",
        )

        # Define the protocols to use for each component, creating a replicator that
        # will copy these for each component in the full substance.
        component_replicator = ProtocolReplicator("component_replicator")
        component_replicator.template_values = ProtocolPath("components", "global")

        component_protocols, component_data_replicator = generate_reweighting_protocols(
            observable_type,
            f"_component_{component_replicator.placeholder_id}",
            f"component_{component_replicator.placeholder_id}_data_replicator",
        )
        component_protocols.reweight_observable.required_effective_samples = (
            n_effective_samples
        )
        component_data_replicator.template_values = ProtocolPath(
            f"component_data[$({component_replicator.id})]", "global"
        )

        divide_by_component_molecules = miscellaneous.DivideValue(
            f"divide_by_component_{component_replicator.placeholder_id}_molecules"
        )
        divide_by_component_molecules.value = ProtocolPath(
            "value", component_protocols.reweight_observable.id
        )
        (
            divide_by_component_molecules.divisor,
            component_n_molar_molecules,
        ) = cls._n_molecules_divisor(
            ProtocolPath(
                "total_number_of_molecules",
                component_protocols.unpack_stored_data.id.replace(
                    component_data_replicator.placeholder_id, "0"
                ),
            ),
            f"_component_{component_replicator.placeholder_id}",
        )

        # Make sure the protocols point to the correct substance.
        component_substance = ReplicatorValue(component_replicator.id)

        component_protocols.build_reference_system.substance = component_substance
        component_protocols.build_target_system.substance = component_substance

        # Weight the component value by the mole fraction.
        weight_by_mole_fraction = miscellaneous.WeightByMoleFraction(
            f"weight_by_mole_fraction_{component_replicator.placeholder_id}"
        )
        weight_by_mole_fraction.value = ProtocolPath(
            "result", divide_by_component_molecules.id
        )
        weight_by_mole_fraction.full_substance = ProtocolPath("substance", "global")
        weight_by_mole_fraction.component = component_substance

        # Finally, set up the protocols which will be responsible for adding together
        # the component observables, and subtracting these from the full system value.
        add_component_observables = miscellaneous.AddValues("add_component_observables")
        add_component_observables.values = ProtocolPath(
            "weighted_value",
            weight_by_mole_fraction.id,
        )

        calculate_excess_observable = miscellaneous.SubtractValues(
            "calculate_excess_observable"
        )
        calculate_excess_observable.value_b = ProtocolPath(
            "value", mixture_protocols.reweight_observable.id
        )
        calculate_excess_observable.value_a = ProtocolPath(
            "result", add_component_observables.id
        )

        # Build the final workflow schema
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            *[x.schema for x in mixture_protocols if x is not None],
            divide_by_mixture_molecules.schema,
            *[x.schema for x in component_protocols if x is not None],
            divide_by_component_molecules.schema,
            weight_by_mole_fraction.schema,
            add_component_observables.schema,
            calculate_excess_observable.schema,
        ]

        if component_n_molar_molecules is not None:
            schema.protocol_schemas.append(component_n_molar_molecules.schema)
        if mixture_n_molar_molecules is not None:
            schema.protocol_schemas.append(mixture_n_molar_molecules.schema)

        schema.protocol_replicators = [
            mixture_data_replicator,
            component_replicator,
            component_data_replicator,
        ]

        schema.final_value_source = ProtocolPath(
            "result", calculate_excess_observable.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @classmethod
    def default_reweighting_schema(
        cls,
        absolute_tolerance: unit.Quantity = UNDEFINED,
        relative_tolerance: float = UNDEFINED,
        n_effective_samples: int = 50,
    ) -> ReweightingSchema:
        """Returns the default calculation schema to use when estimating this class of
        property by re-weighting cached simulation data.

        Parameters
        ----------
        absolute_tolerance
            The absolute tolerance to estimate the property to within.
        relative_tolerance
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_effective_samples
            The minimum number of effective samples to require when
            reweighting the cached simulation data.

        Returns
        -------
            The default re-weighting calculation schema.
        """

        return cls._default_reweighting_schema(
            cls._observable_type(),
            absolute_tolerance,
            relative_tolerance,
            n_effective_samples,
        )
