.. |workflow|           replace:: :py:class:`~evaluator.workflow.Workflow`
.. |workflow_graph|     replace:: :py:class:`~evaluator.workflow.WorkflowGraph`

Workflow Graphs
===============

A |workflow_graph| is a collection of |workflow| objects which should be executed together. The primary advantage of
executing workflows via the graph object is that the graph will automatically take advantage of the :doc:`protocols
<protocols>` built in redundancy / merging support to collapse duplicate tasks across multiple workflows.

As an example, consider the case of executing workflows to estimate the density and the dielectric constant at the
same state point, for the same substance, and using the same force field parameters::

    density_schema = Density.default_simulation_schema()
    dielectric_schema = DielectricConstant.default_simulation_schema()

    density_workflow = Workflow.from_schema(density_schema, metadata)
    dielectric_workflow = Workflow.from_schema(dielectric_schema, metadata)

    print(len(density_workflow.protocols), len(dielectric_workflow.protocols))

    workflow_graph = WorkflowGraph()
    workflow_graph.add_workflow(density_workflow)
    workflow_graph.add_workflow(dielectric_workflow)

    print(len(workflow_graph.protocols))

The final workflow graph has roughly half the total number of density and dielectric protocols to be executed. This
is expected as both the density and dielectric workflows are almost identical, except for the final analysis steps.

Graphs can be executed either in place without using a calculation backend in the same way that :doc:`workflows can
<workflows>`.