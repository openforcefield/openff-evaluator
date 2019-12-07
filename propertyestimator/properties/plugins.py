"""
An API for allowing new properties to be estimated.
"""


registered_properties = {}


def register_estimable_property():
    """A decorator which registers a property as being estimable
    by the property estimator.

    Notes
    -----
    The property must implement a static get_calculation_template method
    which returns the calculation schema to follow.
    """

    def decorator(cls):

        if cls.__name__ in registered_properties:
            raise ValueError(
                "The {} property is already registered.".format(cls.__name__)
            )

        registered_properties[cls.__name__] = cls
        return cls

    return decorator
