.. |protocol_group|           replace:: :py:class:`~openff.evaluator.workflow.ProtocolGroup`

.. |condition|                replace:: :py:class:`~openff.evaluator.protocols.groups.ConditionalGroup.Condition`
.. |conditional_group|        replace:: :py:class:`~openff.evaluator.protocols.groups.ConditionalGroup`

.. |protocol_schema|          replace:: :py:class:`~openff.evaluator.workflow.schemas.ProtocolSchema`
.. |protocol_group_schema|    replace:: :py:class:`~openff.evaluator.workflow.schemas.ProtocolGroupSchema`

.. |protocol_path|            replace:: :py:class:`~openff.evaluator.workflow.utils.ProtocolPath`

.. |current_iteration|        replace:: :py:attr:`~openff.evaluator.protocols.groups.ConditionalGroup.current_iteration`
.. |max_iterations|           replace:: :py:attr:`~openff.evaluator.protocols.groups.ConditionalGroup.max_iterations`

Protocol Groups
===============

The |protocol_group| class represents a collection of :doc:`protocols <protocols>` which have been grouped together. All
protocols within a group will be executed together on a single compute resources, i.e. there is currently no support for
executing protocols within a group in parallel.

Protocol groups have a specialised |protocol_group_schema| which is essentially a collection of |protocol_schema|
objects.

Conditional Protocol Groups
---------------------------

A |conditional_group| is a special class of |protocol_group| which will execute all of the grouped protocols again
and again until a set of conditions has been met or until a maximum number of iterations (see |max_iterations|) has been
performed. They can be thought of as being a protocol representation of a ``while`` statement.

Each condition to be met is represented by a |condition| object::

    condition = ConditionalGroup.Condition()

    # Set the left and right hand values.
    condition.left_hand_value = ...
    condition.right_hand_value = ...

    # Choose the type of condition
    condition.type = ConditionalGroup.Condition.Type.LessThan

The left and right hand values can either be constants, or come from the output of another protocol (including grouped
protocols) using a |protocol_path|. Currently a condition can either check that a value is less than or greater than
another value.

Conditional groups expose a |current_iteration| attribute which tracks how many times the grouped protocols have been
executed. This can be used as input by any of the grouped protocols and is useful, for example, to run a simulation for
longer and longer until the groups condition has been met::

    conditional_group = ConditionalGroup("conditional_group")

    # Set up protocols to run a simulation and then to extract the
    # value of the density and its uncertainty.
    simulation = OpenMMSimulation("simulation")
    simulation.input_coordinate_file = "coords.pdb"
    simulation.system_path = "system.xml"

    extract_density = ExtractAverageStatistic("extract_density")
    extract_density.statistics_type = ObservableType.Density
    extract_density.statistics_path = simulation.statistics_file_path

    # Set the total number of iterations the simulation should perform to be equal
    # to the current iteration of the group. I.e the simulation should perform a
    # new iteration at each group iteration.
    simulation.total_number_of_iterations = ProtocolPath(
        "current_iteration", conditional_group.id
    )

    # Add the protocols to the group.
    conditional_group.add_protocols(production_simulation, analysis_protocol)

    # Set up a condition which will check if the uncertainty is less than
    # some threshold.
    condition = ConditionalGroup.Condition()
    condition.condition_type = groups.ConditionalGroup.Condition.Type.LessThan

    condition.right_hand_value = 0.5 * unit.gram / unit.millilitre
    condition.left_hand_value = ProtocolPath(
        "value.error", conditional_group.id, analysis_protocol.id
    )

    # Add the condition.
    conditional_group.add_condition(condition)

It is this idea which is used to continue running a molecular simulations until an observable of interest (such as the
density) has been calculated to within a specified uncertainty.
