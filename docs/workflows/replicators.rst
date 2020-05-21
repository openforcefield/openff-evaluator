.. |protocol|           replace:: :py:class:`~openff.evaluator.workflow.Protocol`
.. |protocol_schema|    replace:: :py:class:`~openff.evaluator.workflow.schemas.ProtocolSchema`
.. |protocol_graph|     replace:: :py:class:`~openff.evaluator.workflow.ProtocolGraph`
.. |protocol_path|      replace:: :py:class:`~openff.evaluator.workflow.utils.ProtocolPath`
.. |workflow|           replace:: :py:class:`~openff.evaluator.workflow.Workflow`
.. |workflow_schema|    replace:: :py:class:`~openff.evaluator.workflow.schemas.WorkflowSchema`
.. |workflow_graph|     replace:: :py:class:`~openff.evaluator.workflow.WorkflowGraph`
.. |workflow_result|    replace:: :py:class:`~openff.evaluator.workflow.WorkflowResult`

.. |generate_default_metadata|    replace:: :py:meth:`~openff.evaluator.workflow.Workflow.generate_default_metadata`

.. |substance|                    replace:: :py:class:`~openff.evaluator.substances.Substance`
.. |thermodynamic_state|          replace:: :py:class:`~openff.evaluator.thermodynamics.ThermodynamicState`

.. |parameter_gradient_key|       replace:: :py:class:`~openff.evaluator.forcefield.ParameterGradientKey`

.. |build_coordinates_packmol|    replace:: :py:class:`~openff.evaluator.protocols.coordinates.BuildCoordinatesPackmol`
.. |build_smirnoff_system|        replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildSmirnoffSystem`

.. |protocol_schemas|             replace:: :py:attr:`~openff.evaluator.workflow.schemas.WorkflowSchema.protocol_schemas`
.. |final_value_source|           replace:: :py:attr:`~openff.evaluator.workflow.schemas.WorkflowSchema.final_value_source`
.. |gradients_sources|            replace:: :py:attr:`~openff.evaluator.workflow.schemas.WorkflowSchema.gradients_sources`
.. |outputs_to_store|             replace:: :py:attr:`~openff.evaluator.workflow.schemas.WorkflowSchema.outputs_to_store`
.. |protocol_replicators|         replace:: :py:attr:`~openff.evaluator.workflow.schemas.WorkflowSchema.protocol_replicators`

.. |result_value|                 replace:: :py:attr:`~openff.evaluator.workflow.WorkflowResult.value`
.. |result_gradients|             replace:: :py:attr:`~openff.evaluator.workflow.WorkflowResult.gradients`
.. |result_data_to_store|         replace:: :py:attr:`~openff.evaluator.workflow.WorkflowResult.data_to_store`

.. |property_name|                replace:: :py:attr:`~openff.evaluator.workflow.utils.ProtocolPath.property_name`

.. |protocol_replicator|          replace:: :py:class:`~openff.evaluator.workflow.schemas.ProtocolReplicator`
.. |replicator_value|             replace:: :py:class:`~openff.evaluator.workflow.utils.ReplicatorValue`
.. |placeholder_id|               replace:: :py:attr:`~openff.evaluator.workflow.schemas.ProtocolReplicator.placeholder_id`

.. |quantity|                     replace:: :py:class:`~pint.Quantity`

Replicators
===========

A |protocol_replicator| is the workflow equivalent of a ``for`` loop. It is statically evaluated when a |workflow| is
created from its schema. This is useful when parts of a workflow should be run multiple times but using different
values for certain protocol inputs.

.. note:: The syntax of replicators is still rather rough around the edges, and will be refined in future versions of
          the framework.

Each |protocol_replicator| requires both a unique id and the set of *template values* which the replicator will 'loop'
over to be defined. These values must either be a list of constant values or a reference to a list of values provided
as *metadata*.

The 'loop variable' is referenced by protocols in the workflow using the |replicator_value| placeholder input,
where the value is linked to the replicator through the replicators unique id.

As an example, consider the case where a set of coordinates should be built for each component in a substance::

    # Create the replicator object, and assign it a unique id.
    replicator = ProtocolReplicator(replicator_id="component_replicator")
    # Instruct the replicator to loop over all of the components of the substance
    # made available by the global metadata
    replicator.template_values = ProtocolPath("substance.components", "global")

    # Define a protocol which will build some coordinates for a system.
    build_coords = BuildCoordinatesPackmol("build_coords_" + replicator.placeholder_id})
    # Instruct the protocol to use the value specified by the replicator.
    build_coords.substance = ReplicatorValue(replicator.id)

    # Build the schema containing the protocol and the replicator
    schema = WorkflowSchema()
    schema.protocol_schemas = [build_coords.schema]
    schema.protocol_replicators = [replicator]

The requirement for a protocol to be replicated by a replicator is that its id *must* contain the replicators
|placeholder_id| - this is a simple string which the workflow engine looks for when applying the replicator. The
contents of this schema can be easily inspected by printing its JSON representation:

.. code-block:: json

    {
        "@type": "openff.evaluator.workflow.schemas.WorkflowSchema",
        "protocol_replicators": [
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolReplicator",
                "id": "component_replicator",
                "template_values": {
                    "@type": "openff.evaluator.workflow.utils.ProtocolPath",
                    "full_path": "global.substance.components"
                }
            }
        ],
        "protocol_schemas": [
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolSchema",
                "id": "build_coords_$(component_replicator)",
                "inputs": {
                    ".substance": {
                        "@type": "openff.evaluator.workflow.utils.ReplicatorValue",
                        "replicator_id": "component_replicator"
                    }
                },
                "type": "BuildCoordinatesPackmol"
            }
        ]
    }

It can be clearly seen that the schema only contains a single protocol entry, with the placeholder id present in its
unique id. Once a workflow is created from this schema however::

    # Define some metadata
    metadata = {"substance": Substance.from_components("O", "CO")}

    # Build the workflow from the schema.
    workflow = Workflow.from_schema(schema, metadata)
    # Output the contents of the workflow as JSON.
    print(workflow.schema.json())

it can be seen that the replicator has been correctly been applied and the workflow now contains one protocol for each
component in the substance passed as metadata:

.. code-block:: json

    {
        "@type": "openff.evaluator.workflow.schemas.WorkflowSchema",
        "protocol_schemas": [
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolSchema",
                "id": "build_coords_0",
                "inputs": {
                    ".substance": {
                        "@type": "openff.evaluator.substances.components.Component",
                        "smiles": "O"
                    }
                },
                "type": "BuildCoordinatesPackmol"
            },
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolSchema",
                "id": "build_coords_1",
                "inputs": {
                    ".substance": {
                        "@type": "openff.evaluator.substances.components.Component",
                        "smiles": "CO"
                    }
                },
                "type": "BuildCoordinatesPackmol"
            }
        ]
    }

In both cases the replicators |placeholder_id| has been replaced with the index of the value it was replicated for, and
the substance input has been correctly set to the actual array value.

Nested Replicators
------------------

Replicators can be applied to other replicators to achieve a result similar to a set of nested for loops. For example
the below loop::

    components = [Component("O"), Component("CO")]
    n_mols = [[1000], [500]]

    for i, component in enumerate(components):

        for component_n_mols in n_mols[i]:

            ...

can readily be reproduced using replicators::

    # Define a replicator which will loop over all components in the substance.
    component_replicator = ProtocolReplicator(replicator_id="components")
    component_replicator.template_values = ProtocolPath("components", "global")

    # Define a replicator to loop over the number of each component to add.
    n_mols_replicator_id = f"n_mols_{component_replicator.placeholder_id}"

    n_mols_replicator = ProtocolReplicator(replicator_id=n_mols_replicator_id)
    n_mols_replicator.template_values = ProtocolPath(
        f"n_mols[{component_replicator.placeholder_id}]", "global"
    )

    # Define the suffix which must be applied to protocols to be replicated
    id_suffix = f"{component_replicator.placeholder_id}_{n_mols_replicator.placeholder_id}"

    # Define a protocol which will build some coordinates for a system.
    build_coordinates = BuildCoordinatesPackmol(f"build_coordinates_{id_suffix}")
    build_coordinates.substance = ReplicatorValue(component_replicator.id)
    build_coordinates.max_molecules = ReplicatorValue(n_mols_replicator.id)

    # Build the schema containing the protocol and the replicator
    schema = WorkflowSchema()
    schema.protocol_schemas = [build_coordinates.schema]
    schema.protocol_replicators = [component_replicator, n_mols_replicator]

    # Define some metadata
    metadata = {
        "components": [Component("O"), Component("CO")],
        "n_mols": [[1000], [500]]
    }

    # Build the workflow from the created schema.
    workflow = Workflow.from_schema(schema, metadata)
    # Print the JSON representation of the workflow.
    print(workflow.schema.json(format=True))

Here the ``component_replicator`` placeholder id has been appended to the ``n_mols_replicator`` id to inform the
workflow engine that the later is a child of the former. The ``component_replicator`` placeholder id is then used
as an index into the ``n_mols`` array. This results in the following schema as desired:

.. code-block:: json

    {
        "@type": "openff.evaluator.workflow.schemas.WorkflowSchema",
        "protocol_schemas": [
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolSchema",
                "id": "build_coordinates_0_0",
                "inputs": {
                    ".max_molecules": 1000,
                    ".substance": {
                        "@type": "openff.evaluator.substances.components.Component",
                        "smiles": "O"
                    }
                },
                "type": "BuildCoordinatesPackmol"
            },
            {
                "@type": "openff.evaluator.workflow.schemas.ProtocolSchema",
                "id": "build_coordinates_1_0",
                "inputs": {
                    ".max_molecules": 500,
                    ".substance": {
                        "@type": "openff.evaluator.substances.components.Component",
                        "smiles": "CO"
                    }
                },
                "type": "BuildCoordinatesPackmol"
            }
        ]
    }
