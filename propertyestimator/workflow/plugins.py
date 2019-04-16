"""
An API for allowing new workflow protocols to be used in property estimations.
"""

available_protocols = {}


def register_calculation_protocol():
    """A decorator which registers a class as being a
    protocol which may be used in calculation schemas.
    """

    def decorator(cls):

        if cls.__name__ in available_protocols:
            raise ValueError('The {} protocol is already registered.'.format(cls.__name__))

        available_protocols[cls.__name__] = cls
        return cls

    return decorator
