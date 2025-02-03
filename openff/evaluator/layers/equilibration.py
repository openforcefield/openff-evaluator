"""A calculation layer for equilibration.
"""

import copy
import logging
import os

from openff.units import unit

from openff.evaluator.attributes import (
    UNDEFINED,
    Attribute,
    AttributeClass,
    PlaceholderValue
)
from openff.evaluator.layers import calculation_layer
from openff.evaluator.layers.layers import BaseCalculationLayerSchema
from openff.evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
    BaseWorkflowCalculationSchema,
    WorkflowGraph
)
from openff.evaluator.datasets import CalculationSource, PropertyPhase
from openff.evaluator.workflow.attributes import ConditionAggregationBehavior
from openff.evaluator.utils.observables import ObservableType, ObservableFrame
from openff.evaluator.workflow import Workflow
from openff.evaluator.storage.query import EquilibrationDataQuery

logger = logging.getLogger(__name__)

def default_storage_query():
    """Return the default query to use when retrieving cached simulation
     data from the storage backend.
    Currently this query will search for data for the full substance of
    interest in the liquid phase.
    Returns
    -------
    dict of str and SimulationDataQuery
        A single query with a key of `"full_system_data"`.
    """

    query = EquilibrationDataQuery()
    query.substance = PlaceholderValue()
    query.thermodynamic_state = PlaceholderValue()
    query.max_number_of_molecules = PlaceholderValue()

    query.property_phase = PropertyPhase.Liquid
    query.calculation_layer = "EquilibrationLayer"

    return {"full_system_data": query}

class EquilibrationProperty(AttributeClass):
    """A schema which encodes the options that a `CalculationLayer`
    should use when estimating a given class of physical properties.
    """

    absolute_tolerance = Attribute(
        docstring="The absolute uncertainty that the property should "
        "be estimated to within. This attribute is mutually exclusive "
        "with the `relative_tolerance` attribute.",
        type_hint=unit.Quantity,
        default_value=UNDEFINED,
        optional=True,
    )
    relative_tolerance = Attribute(
        docstring="The relative uncertainty that the property should "
        "be estimated to within, i.e `relative_tolerance * "
        "measured_property`. This attribute is mutually "
        "exclusive with the `absolute_tolerance` attribute.",
        type_hint=float,
        default_value=UNDEFINED,
        optional=True,
    )
    observable_type = Attribute(
        docstring="The type of observable to use in evaluating equilibration.",
        type_hint=ObservableType,
        optional=False
    )
    n_uncorrelated_samples = Attribute(
        docstring="The number of uncorrelated samples to use in evaluating equilibration.",
        type_hint=int,
        default_value=UNDEFINED,
        optional=True
    )

    def validate(self, attribute_type=None):
        if (
            self.absolute_tolerance != UNDEFINED
            and self.relative_tolerance != UNDEFINED
        ):
            raise ValueError(
                "Only one of `absolute_tolerance` and `relative_tolerance` "
                "can be set."
            )

        # check units
        if self.absolute_tolerance != UNDEFINED:
            assert self.absolute_tolerance.units.is_compatible_with(self.observable_unit)
        super(EquilibrationProperty, self).validate(attribute_type)

    @property
    def observable_unit(self):
        return ObservableFrame._units[self.observable_type]

    @property
    def tolerance(self):
        if self.absolute_tolerance != UNDEFINED:
            return self.absolute_tolerance
        return self.relative_tolerance



class EquilibrationSchema(WorkflowCalculationSchema):
    """
    A schema which encodes the options that a `EquilibrationLayer`
    should use when equilibrating a given box.

    Analogous to the `CalculationLayerSchema` or `WorkflowCalculationSchema` class for normal properties.
    """

    error_tolerances = Attribute(
        docstring="The error tolerances to use when equilibrating the box.",
        type_hint=list,
        default_value=[],
    )
    error_aggregration = Attribute(
        docstring="How to aggregate errors -- any vs all.",
        type_hint=ConditionAggregationBehavior,
        default_value=ConditionAggregationBehavior.All,
    )
    error_on_failure = Attribute(
        docstring="Whether to raise an error if the convergence conditions are not met.",
        type_hint=bool,
        default_value=True,
    )
    max_iterations = Attribute(
        docstring="The maximum number of iterations to run the equilibration for.",
        type_hint=int,
        default_value=100,
    )
    storage_queries = Attribute(
        docstring="The queries to perform when retrieving data for each "
        "of the components in the system from the storage backend. The "
        "keys of this dictionary will correspond to the metadata keys made "
        "available to the workflow system.",
        type_hint=dict,
        default_value=default_storage_query(),
    )
    number_of_molecules = Attribute(
        docstring="The number of molecules in the system.",
        type_hint=int,
    )

    def validate(self, attribute_type=None):
        if self.error_tolerances:
            for error_tolerance in self.error_tolerances:
                assert isinstance(error_tolerance, EquilibrationProperty)
                error_tolerance.validate()
        super(EquilibrationSchema, self).validate(attribute_type)



@calculation_layer()
class EquilibrationLayer(WorkflowCalculationLayer):
    """A calculation layer which employs molecular simulation
    to estimate sets of physical properties.
    """

    @staticmethod
    def _get_workflow_metadata(
        working_directory,
        physical_property,
        force_field_path,
        parameter_gradient_keys,
        storage_backend,
        calculation_schema,
    ):
        """Returns the global metadata to pass to the workflow.

        Parameters
        ----------
        working_directory: str
            The local directory in which to store all local,
            temporary calculation data from this workflow.
        physical_property : PhysicalProperty
            The property that the workflow will estimate.
        force_field_path : str
            The path to the force field parameters to use in the workflow.
        parameter_gradient_keys: list of ParameterGradientKey
            A list of references to all of the parameters which all observables
            should be differentiated with respect to.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        calculation_schema: EquilibrationLayerSchema
            The schema containing all of this layers options.

        Returns
        -------
        dict of str and Any, optional
            The global metadata to make available to a workflow.
            Returns `None` if the required metadata could not be
            found / assembled.
        """

        global_metadata = Workflow.generate_default_metadata(
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            None,  # set target to None for now
        )
        global_metadata["error_tolerances"] = copy.deepcopy(calculation_schema.error_tolerances)
        global_metadata["error_aggregation"] = calculation_schema.error_aggregration

        # search storage for matching boxes already
        template_queries = calculation_schema.storage_queries
        for key in template_queries:
            query = EquilibrationLayer._update_query(
                template_queries[key],
                physical_property,
                calculation_schema,
            )

            # Apply the query.
            query_results = storage_backend.query(query)

            stored_data_tuples = []
            for query_list in query_results.values():
                for storage_key, data_object, data_directory in query_list:
                    object_path = os.path.join(working_directory, f"{storage_key}.json")
                    if os.path.isfile(object_path):
                        stored_data_tuples.append((object_path, data_directory, force_field_path))
                    
            if len(stored_data_tuples) == 1:
                stored_data_tuples = stored_data_tuples[0]
            
            global_metadata[key] = stored_data_tuples

        return global_metadata
    

    @staticmethod
    def _update_query(query, physical_property, calculation_schema):
        query = copy.deepcopy(query)

        # Fill in any place holder values.
        if isinstance(query.thermodynamic_state, PlaceholderValue):
            query.thermodynamic_state = physical_property.thermodynamic_state
        if isinstance(query.max_number_of_molecules, PlaceholderValue):
            query.max_number_of_molecules = calculation_schema.number_of_molecules

        # need to treat the substance specially as mole fractions can vary with number of molecules
        if isinstance(query.substance, PlaceholderValue):
            query.substance = physical_property.substance.to_substance_n_molecules(
                calculation_schema.number_of_molecules
            )
            # query.substance = physical_property.substance
        return query
    



    @classmethod
    def required_schema_type(cls):
        return EquilibrationSchema
