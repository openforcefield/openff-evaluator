"""Provides base classes for calculation layers which will
use the built-in workflow framework to estimate the set of
physical properties.
"""
import abc
import copy
import logging
import os

from openff.evaluator.attributes import UNDEFINED, Attribute
from openff.evaluator.datasets import CalculationSource
from openff.evaluator.layers import (
    CalculationLayer,
    CalculationLayerResult,
    CalculationLayerSchema,
)
from openff.evaluator.workflow import Workflow, WorkflowGraph, WorkflowSchema

logger = logging.getLogger(__name__)


class WorkflowCalculationLayer(CalculationLayer, abc.ABC):
    """An calculation layer which uses the built-in workflow
    framework to estimate sets of physical properties.
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
        calculation_schema: WorkflowCalculationSchema
            The schema containing all of this layers options.

        Returns
        -------
        dict of str and Any, optional
            The global metadata to make available to a workflow.
            Returns `None` if the required metadata could not be
            found / assembled.
        """
        target_uncertainty = None

        if calculation_schema.absolute_tolerance != UNDEFINED:
            target_uncertainty = calculation_schema.absolute_tolerance
        elif calculation_schema.relative_tolerance != UNDEFINED:
            target_uncertainty = (
                physical_property.uncertainty * calculation_schema.relative_tolerance
            )

        global_metadata = Workflow.generate_default_metadata(
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            target_uncertainty,
        )

        return global_metadata

    @classmethod
    def _build_workflow_graph(
        cls,
        working_directory,
        storage_backend,
        properties,
        force_field_path,
        parameter_gradient_keys,
        options,
    ):
        """Construct a graph of the protocols needed to calculate a set of
        properties.

        Parameters
        ----------
        working_directory: str
            The local directory in which to store all local,
            temporary calculation data from this graph.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        properties : list of PhysicalProperty
            The properties to attempt to compute.
        force_field_path : str
            The path to the force field parameters to use in the workflow.
        parameter_gradient_keys: list of ParameterGradientKey
            A list of references to all of the parameters which all observables
            should be differentiated with respect to.
        options: RequestOptions
            The options to run the workflows with.
        """

        provenance = {}
        workflows = []

        for index, physical_property in enumerate(properties):

            logger.info(f"Building workflow {index} of {len(properties)}")

            property_type = type(physical_property).__name__

            # Make sure a schema has been defined for this class of property
            # and this layer.
            if (
                property_type not in options.calculation_schemas
                or cls.__name__ not in options.calculation_schemas[property_type]
            ):
                continue

            schema = options.calculation_schemas[property_type][cls.__name__]

            # Make sure the calculation schema is the correct type for this layer.
            assert isinstance(schema, WorkflowCalculationSchema)
            assert isinstance(schema, cls.required_schema_type())

            global_metadata = cls._get_workflow_metadata(
                working_directory,
                physical_property,
                force_field_path,
                parameter_gradient_keys,
                storage_backend,
                schema,
            )

            if global_metadata is None:
                # Make sure we have metadata returned for this
                # property, e.g. we have data to reweight if
                # required.
                continue

            workflow = Workflow(global_metadata, physical_property.id)
            workflow.schema = schema.workflow_schema
            workflows.append(workflow)

        workflow_graph = WorkflowGraph()
        workflow_graph.add_workflows(*workflows)

        for workflow in workflows:

            provenance[workflow.uuid] = CalculationSource(
                fidelity=cls.__name__, provenance=workflow.schema.json()
            )

        return workflow_graph, provenance

    @staticmethod
    def workflow_to_layer_result(queued_properties, provenance, workflow_results, **_):
        """Converts a list of `WorkflowResult` to a list of `CalculationLayerResult`
        objects.

        Parameters
        ----------
        queued_properties: list of PhysicalProperty
            The properties being estimated by this layer
        provenance: dict of str and str
            The provenance of each property.
        workflow_results: list of WorkflowResult
            The results of each workflow.

        Returns
        -------
        list of CalculationLayerResult
            The calculation layer result objects.
        """
        properties_by_id = {x.id: x for x in queued_properties}
        results = []

        for workflow_result in workflow_results:

            calculation_result = CalculationLayerResult()
            calculation_result.exceptions.extend(workflow_result.exceptions)

            results.append(calculation_result)

            if len(calculation_result.exceptions) > 0:
                continue

            physical_property = properties_by_id[workflow_result.workflow_id]
            physical_property = copy.deepcopy(physical_property)
            physical_property.source = provenance[physical_property.id]
            physical_property.value = workflow_result.value.value
            physical_property.uncertainty = workflow_result.value.error

            if len(workflow_result.gradients) > 0:
                physical_property.gradients = workflow_result.gradients

            calculation_result.physical_property = physical_property
            calculation_result.data_to_store.extend(workflow_result.data_to_store)

        return results

    @classmethod
    def _schedule_calculation(
        cls,
        calculation_backend,
        storage_backend,
        layer_directory,
        batch,
    ):

        # Store a temporary copy of the force field for protocols to easily access.
        force_field_source = storage_backend.retrieve_force_field(batch.force_field_id)
        force_field_path = os.path.join(layer_directory, batch.force_field_id)

        with open(force_field_path, "w") as file:
            file.write(force_field_source.json())

        workflow_graph, provenance = cls._build_workflow_graph(
            layer_directory,
            storage_backend,
            batch.queued_properties,
            force_field_path,
            batch.parameter_gradient_keys,
            batch.options,
        )

        workflow_futures = workflow_graph.execute(layer_directory, calculation_backend)

        future = calculation_backend.submit_task(
            WorkflowCalculationLayer.workflow_to_layer_result,
            batch.queued_properties,
            provenance,
            workflow_futures,
        )

        return [future]


class WorkflowCalculationSchema(CalculationLayerSchema):
    """A schema which encodes the options and the workflow schema
    that a `CalculationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    workflow_schema = Attribute(
        docstring="The workflow schema to use when estimating properties.",
        type_hint=WorkflowSchema,
        default_value=UNDEFINED,
    )

    def validate(self, attribute_type=None):
        super(WorkflowCalculationSchema, self).validate(attribute_type)
        self.workflow_schema.validate()
