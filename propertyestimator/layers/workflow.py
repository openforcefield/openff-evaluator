"""Provides base classes for calculation layers which will
use the built-in workflow framework to estimate the set of
physical properties.
"""
import abc
import os

from propertyestimator.attributes import UNDEFINED, Attribute
from propertyestimator.datasets import CalculationSource
from propertyestimator.layers import CalculationLayer, CalculationLayerSchema
from propertyestimator.workflow import Workflow, WorkflowGraph, WorkflowSchema


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

        global_metadata = Workflow.generate_default_metadata(
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            calculation_schema.workflow_options,
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
        """ Construct a graph of the protocols needed to calculate a set of
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
        workflow_graph = WorkflowGraph(working_directory)

        for physical_property in properties:

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

            workflow = Workflow(physical_property, global_metadata)
            workflow.schema = schema.workflow_schema

            workflow.physical_property.source = CalculationSource(
                fidelity=cls.__name__, provenance={}
            )

            workflow_graph.add_workflow(workflow)

        return workflow_graph

    @classmethod
    def schedule_calculation(
        cls,
        calculation_backend,
        storage_backend,
        layer_directory,
        batch,
        callback,
        synchronous=False,
    ):

        # Store a temporary copy of the force field for protocols to easily access.
        force_field_source = storage_backend.retrieve_force_field(batch.force_field_id)
        force_field_path = os.path.join(layer_directory, batch.force_field_id)

        with open(force_field_path, "w") as file:
            file.write(force_field_source.json())

        workflow_graph = cls._build_workflow_graph(
            layer_directory,
            storage_backend,
            batch.queued_properties,
            force_field_path,
            batch.parameter_gradient_keys,
            batch.options,
        )

        futures = workflow_graph.submit(calculation_backend)

        CalculationLayer._await_results(
            calculation_backend, storage_backend, batch, callback, futures, synchronous,
        )


class WorkflowCalculationSchema(CalculationLayerSchema):
    """A schema which encodes the options and the workflow schema
    that a `CalculationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    # TODO: Implement.
    allow_protocol_merging = Attribute(
        docstring="If `True`, the workflow framework will attempt to merge "
        "redundant protocols from multiple workflow graphs into "
        "a single protocol execution.",
        type_hint=bool,
        default_value=True,
    )

    workflow_schema = Attribute(
        docstring="The workflow schema to use when estimating properties.",
        type_hint=WorkflowSchema,
        default_value=UNDEFINED,
    )

    def validate(self, attribute_type=None):
        super(WorkflowCalculationSchema, self).validate(attribute_type)
        self.workflow_schema.validate()
