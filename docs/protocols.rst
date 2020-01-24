.. |protocol|             replace:: :py:class:`~propertyestimator.workflow.Protocol`
.. |protocol_schema|      replace:: :py:class:`~propertyestimator.workflow.schemas.ProtocolSchema`
.. |protocol_graph|       replace:: :py:class:`~propertyestimator.workflow.ProtocolGraph`
.. |protocol_path|        replace:: :py:class:`~propertyestimator.workflow.utils.ProtocolPath`
.. |workflow|             replace:: :py:class:`~propertyestimator.workflow.Workflow`
.. |workflow_schema|      replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema`
.. |workflow_graph|       replace:: :py:class:`~propertyestimator.workflow.WorkflowGraph`
.. |workflow_result|      replace:: :py:class:`~propertyestimator.workflow.WorkflowResult`

.. |input_attribute|      replace:: :py:class:`~propertyestimator.workflow.attributes.InputAttribute`
.. |output_attribute|     replace:: :py:class:`~propertyestimator.workflow.attributes.OutputAttribute`
.. |undefined|            replace:: :py:class:`~propertyestimator.attributes.UNDEFINED`

.. |openmm_simulation|    replace:: :py:class:`~propertyestimator.protocols.openmm.OpenMMSimulation`

.. |compute_resources|    replace:: :py:class:`~propertyestimator.backends.ComputeResources`

Protocols
=========

The |protocol| class represents a single task to be executed, whether that be as a standalone task or as a task which is
part of some larger workflow. The task encoded by a protocol may be as simple as adding two numbers together or even
as complex as performing entire free energy simulations::

    from propertyestimator.protocols.miscellaneous import AddValues

    # Create the protocol and assign it some unique name.
    add_numbers = AddValues(protocol_id="add_values")
    # Set the numbers to add together
    add_numbers.values = [1, 2, 3, 4]

    # Execute the protocol
    add_numbers.execute()

    # Retrieve the output
    add_value = add_numbers.result

Each protocol exposes a set of the required inputs as well as the produced outputs. These inputs may either be set as a
constant directly, or if used as part of a :doc:`workflow <workflows>`, can take their value from one of the outputs
of another protocol.

.. figure:: _static/img/protocol.svg
    :align: center
    :width: 250

    A selection of the inputs and outputs of the |openmm_simulation| protocol.

A surprisingly rich spectrum of workflows can be constructed by chaining together many relatively simple protocols.

The inputs and outputs of a protocol are defined using the custom |input_attribute| and |output_attribute| descriptors::

    class AddValues(Protocol):

        # Define the inputs that the protocol requires
        values = InputAttribute(
            docstring="The values to add together.",
            type_hint=list, default_value=UNDEFINED
        )

        # Define the outputs that the protocol will produce
        # once it is executed.
        result = OutputAttribute(
            docstring="The sum of the values.",
            type_hint=typing.Union[int, float, pint.Measurement, pint.Quantity],
        )

        def _execute(self, directory, available_resources):
            ...

        def validate(self, attribute_type=None):
            ...

Here we have defined a ``values`` input to the protocol and a ``result`` output. Both descriptors require a
``docstring`` and a ``type_hint`` to be provided.

The ``type_hint`` will be used by the workflow engine to ensure that a protocol which takes its input as the output of
another protocol is receiving values of the correct type. Currently the ``type_hint`` can be any type of python class,
or a ``Union`` of multiple types should the protocol allow for that.

In addition, the input attribute must specify a ``default_value`` for the attribute. This can either be a constant
value, or a value set by some function such as a ``lambda`` statement::

    some_input = InputAttribute(
        docstring="Takes it's default value from a function.",
        type_hint=int,
        default_value=lambda: return 1 + 1
    )

In the above example we set the default value of ``values`` to |undefined| in order to specify that this input must be
set by the user. The custom |undefined| class is used in place of ``None`` as ``None`` may be a valid input value for
some attributes.

In addition to defining its inputs and outputs, a protocol must also implement an ``_execute`` method which handles the
main logic of the task::

    def _execute(self, directory, available_resources):

        self.result = self.values[0]

        for value in self.values[1:]:
            self.result += value

The function is passed the directory in which it should run and create any working files, as well as a
|compute_resources| object which describes which compute resources are available to run on. This method *must* set all
of the output attributes of the protocol before returning.

The private ``_execute`` method which must be implemented should not be confused with the public ``execute`` method. The
public ``execute`` method implements some common protocol logic (such as validating the inputs and creating the
directory to run in if needed) before calling the private ``_execute`` method.

The protocols inputs will automatically be validated before ``_execute`` is called - this validation includes making
sure that all of the non-optional inputs have been set, as well as ensuring they have been set to the correct type.
Protocols may implement additional validation logic by implementing a ``validate`` method::

    def validate(self, attribute_type=None):

        super(AddValues, self).validate(attribute_type)

        if len(self.values) < 1:
            raise ValueError("There were no values to add together")
