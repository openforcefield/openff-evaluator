"""
An API for registering new workflow protocols.

Attributes
----------
registered_workflow_protocols: dict of str and type of BaseProtocol
    The workflow protocols which have been registered as being
    available to use in property estimations.
"""
registered_workflow_protocols = {}


def register_workflow_protocol(protocol_class):
    """Registers a class as being a protocol which may be included
    in workflows.
    """
    from propertyestimator.workflow.protocols import BaseProtocol
    assert issubclass(protocol_class, BaseProtocol)

    if protocol_class.__name__ in registered_workflow_protocols:
        raise ValueError(f"The {protocol_class} protocol is already registered.")

    registered_workflow_protocols[protocol_class.__name__] = protocol_class


def workflow_protocol():
    """A decorator which registers a class as being a protocol
    which may be included in workflows.
    """

    def decorator(cls):
        register_workflow_protocol(cls)
        return cls

    return decorator
