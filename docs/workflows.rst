.. |protocol|           replace:: :py:class:`~propertyestimator.workflow.Protocol`
.. |protocol_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.ProtocolSchema`
.. |protocol_graph|     replace:: :py:class:`~propertyestimator.workflow.ProtocolGraph`
.. |protocol_path|      replace:: :py:class:`~propertyestimator.workflow.utils.ProtocolPath`
.. |workflow|           replace:: :py:class:`~propertyestimator.workflow.Workflow`
.. |workflow_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema`
.. |workflow_graph|     replace:: :py:class:`~propertyestimator.workflow.WorkflowGraph`
.. |workflow_result|    replace:: :py:class:`~propertyestimator.workflow.WorkflowResult`

.. |generate_default_metadata|    replace:: :py:meth:`~propertyestimator.workflow.Workflow.generate_default_metadata`

.. |substance|                    replace:: :py:class:`~propertyestimator.substances.Substance`
.. |thermodynamic_state|          replace:: :py:class:`~propertyestimator.thermodynamics.ThermodynamicState`

.. |parameter_gradient_key|       replace:: :py:class:`~propertyestimator.forcefield.ParameterGradientKey`

.. |build_coordinates_packmol|    replace:: :py:class:`~propertyestimator.protocols.coordinates.BuildCoordinatesPackmol`
.. |build_smirnoff_system|        replace:: :py:class:`~propertyestimator.protocols.forcefield.BuildSmirnoffSystem`

.. |protocol_schemas|             replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema.protocol_schemas`
.. |final_value_source|           replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema.final_value_source`
.. |gradients_sources|            replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema.gradients_sources`
.. |outputs_to_store|             replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema.outputs_to_store`
.. |protocol_replicators|         replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema.protocol_replicators`

.. |result_value|                 replace:: :py:class:`~propertyestimator.workflow.WorkflowResult.value`
.. |result_gradients|             replace:: :py:class:`~propertyestimator.workflow.WorkflowResult.gradients`
.. |result_data_to_store|         replace:: :py:class:`~propertyestimator.workflow.WorkflowResult.data_to_store`

.. |property_name|                replace:: :py:class:`~propertyestimator.workflow.utils.ProtocolPath.property_name`

.. |protocol_replicator|          replace:: :py:class:`~propertyestimator.workflow.schemas.ProtocolReplicator`
.. |replicator_value|             replace:: :py:class:`~propertyestimator.workflow.utils.ReplicatorValue`

.. |quantity|                     replace:: :py:class:`~pint.Quantity`

Workflows
=========

The framework offers a lightweight workflow engine for executing graphs of tasks using the available :doc:`calculation
backends <calculationbackend>`. While lightweight, it offers a large amount of extensibility and flexibility, and is
currently used by both the :doc:`simulation <simulationlayer>` and :doc:`reweighting <reweightinglayer>` layers to
perform their required calculations.

A workflow is a wrapper around a collection of tasks that should be executed in succession, and whose outputs should be
made available as the input to others.

.. figure:: _static/img/coords_and_assign.svg
    :align: center
    :width: 70%

    A an example workflow which combines a protocol which will build a set of coordinates for a particular system,
    assign parameters to that system, and then perform an energy minimisation.

Building Workflows
------------------

At its core a workflow must define the tasks which need to be executed, and where the inputs to those tasks should be
sourced from. Each task to be executed is represented by a :doc:`protocol object <protocols>`, with each protocol
requiring a specific set of user specified inputs::

    # Define a protocol which will build some coordinates for a system.
    build_coordinates = BuildCoordinatesPackmol("build_coordinates")
    build_coordinates.max_molecules = 1000
    build_coordinates.mass_density = 1.0 * unit.gram / unit.millilitre
    build_coordinates.substance = Substance.from_components("O", "CO")

    # Define a protocol which will assign force field parameters to the system.
    assign_parameters = BuildSmirnoffSystem(f"assign_parameters")
    assign_parameters.water_model = BuildSmirnoffSystem.WaterModel.TIP3P
    assign_parameters.force_field_path = "openff-1.0.0.offxml"

    # Set the `coordinate_file_path` input of the `assign_parameters` protocol
    # to the `coordinate_file_path` output of the `build_coordinates` protocol.
    assign_parameters.coordinate_file_path = ProtocolPath(
        "coordinate_file_path", build_coordinates.id
    )

The |protocol_path| object is used to reference the output of a protocol to be executed, and will be replaced by the
workflow engine with the actual value once the ``build_coordinates`` has been executed. It is constructed from two
parts:

* the name of the output attribute to reference (passed as the ``property_name`` constructor argument).
* the unique id of the protocol to take the output from.

To turn these tasks into a valid workflow which can be automatically executed, they must first be converted to a
:ref:`workflow schema <Workflow Schemas>`::

    # Create the schema object.
    schema = WorkflowSchema()
    # Add the individual protocol's schema representations to the workflow schema.
    schema.protocol_schemas = [build_coordinates.schema, assign_parameters.schema]

    # Create the executable workflow object from its schema.
    workflow = Workflow.from_schema(schema, metadata=None)

A |workflow| may either be synchronously executed in place yielding a |workflow_result| object directly::

    workflow_result = workflow.execute()

or asynchronously using a calculation backend yielding a ``Future`` like object which will eventually return a
|workflow_result|::

    with DaskLocalCluster() as calculation_backend:
        result_future = workflow.execute(calculation_backend=calculation_backend)

In addition, a workflow may be add to, and executed as part as a larger :doc:`workflow_graphs`.

Workflow Schemas
----------------

A |workflow_schema| is a blueprint from which all |workflow| objects are constructed. It will predominantly define the
tasks which the workflow will include, but may optionally define:

.. rst-class:: spaced-list

    - |final_value_source|: A reference to the protocol output which corresponds to the value of the
      main observable calculated by the workflow.
    - |gradients_sources|: A list of references to the protocol outputs which correspond to the gradients of the
      main observable with respect to a set of force field parameters.
    - |outputs_to_store|: A list of :doc:`data classes <dataclasses>` whose values will be populated from protocol
      outputs.
    - |protocol_replicators|: A set of :ref:`replicators <Replicators>` which are used to flag parts of a workflow which
      should be replicated.

Each of these attributes will control whether the |result_value|, |result_gradients| and |result_data_to_store|
attributes of the |workflow_result| results object will be populated respectively.

Metadata
""""""""

Because a schema is purely a blueprint for a general workflow, it need not define the exact values of all of the inputs
of its constituent tasks. Consider the above example workflow for constructing a set of coordinates and assigning force
field parameters to them. Ideally this one schema could be reused for multiple substances. This is made possible through
a workflows *metadata*.

Each workflow will make available a dictionary of values (termed here *metadata*) which is defined when a |workflow| is
created from its schema. This metadata may be accessed by protocols via a fictitious ``"global"`` protocol whose outputs
map to the ``metadata`` dictionary::

    build_coordinates = BuildCoordinatesPackmol("build_coordinates")
    build_coordinates.substance = ProtocolPath("substance", "global")

    # ...

    substances = [
        Substance.from_components("CO"),
        Substance.from_components("CCO"),
        Substance.from_components("CCCO"),
    ]

    for substance in substances:

        # Define the metadata to make available to the workflow protocols.
        metadata = {"substance": substance}
        # Create the executable workflow object from its schema.
        workflow = Workflow.from_schema(schema, metadata=metadata)

        # Execute the workflow ...

the created workflow will contain the ``build_coordinates`` protocol but with its ``substance`` input set to the value
from the ``metadata`` dictionary.

Replicators
"""""""""""

A |protocol_replicator| is the workflow equivalent of a ``for`` loop which is evaluated when a |workflow| is created
from its schema. This is useful when parts of a workflow should be run multiple times for different values in an array
but using different input values, such as creating a set of coordinates for each component in a substance::

    example.

Each |protocol_replicator| requires both:

* a unique id which will be used when specifying which protocols should be replicated by a given replicator
* the set of *template values* which the replicator will 'loop' over.

The 'replicator value' can be referenced by protocols in the workflow using the |replicator_value| placeholder input,
where the value is linked to the replicator through the replicators unique id.

Replicators can be applied to other replicators to achieve a result similar to a set of nested for loops::

    example.
