"""A set of helpers for statically comparing protocol input and
output types. This is *not* meant to be a means to implement dynamic
typing into the framework, but rather to aid in statically verifying
the protocols input and outputs align correctly.
"""
import typing

_built_in_types_to_use = {
    list: typing.List,
    tuple: typing.Tuple,
    set: typing.Set,
    frozenset: typing.FrozenSet,
    dict: typing.Dict,
}


_supported_generic_types = {typing.Union}


def _is_typing_object(type_object):
    """Checks to see if a type belong to the typing module.

    Parameters
    ----------
    type_object: type or typing._GenericAlias
        The type to check

    Returns
    -------
    bool
        True if `type_object` is a member of `typing`.
    """
    return type_object.__module__ == "typing"


def _is_supported_generic(type_object):
    """Checks to see if a type is a supported `typing._GenericAlias`
    object.

    Parameters
    ----------
    type_object: type or typing._GenericAlias
        The type to check

    Returns
    -------
    bool
        True if `type_object` is a member of `typing`.
    """
    return (
        _is_typing_object(type_object)
        and type_object.__origin__ in _supported_generic_types
    )


def _is_union_type(type_object):
    """Checks if a typing is a `typing.Union` type.

    Parameters
    ----------
    type_object: type or typing._GenericAlias
        The type to check.

    Returns
    -------
    bool
        True if the type is a `typing.Union` type.
    """
    return _is_supported_generic(type_object) and type_object.__origin__ == typing.Union


def is_supported_type(type_object):
    """Returns if the type can be compared using the utilities in this
    class.

    Parameters
    ----------
    type_object: type
        The type to validate.

    Returns
    -------
    bool
        True if the type can be compared using the utilities in this.
    """
    return not _is_typing_object(type_object) or (
        _is_typing_object(type_object) and _is_supported_generic(type_object)
    )


def is_type_subclass_of_type(type_a, type_b):
    """Returns if `type_a` is a subclass of `type_b`.

    Parameters
    ----------
    type_a: type or typing._GenericAlias
        The type to compare.
    type_b: type or typing._GenericAlias
        The subclass type.

    Returns
    -------
    bool
        True if `type_a` is a subclass of `type_b`
    """

    # Make sure these are types we support.
    if _is_typing_object(type_a) and not _is_supported_generic(type_a):

        raise ValueError(
            f'Only the {" ".join(map(str, _supported_generic_types))} '
            f"typing module types are supported, and not {type_a}."
        )

    if _is_typing_object(type_b) and not _is_supported_generic(type_b):

        raise ValueError(
            f'Only the {" ".join(map(str, _supported_generic_types))} '
            f"typing module types are supported, and not {type_b}."
        )

    if _is_union_type(type_a):

        for arg in type_a.__args__:

            if not is_type_subclass_of_type(arg, type_b):
                continue

            return True

        return False

    if _is_union_type(type_b):

        for arg in type_b.__args__:

            if not is_type_subclass_of_type(type_a, arg):
                continue

            return True

        return False

    if type_a == int and type_b == float:
        # All ints can be converted to floats.
        return True

    return issubclass(type_a, type_b)


def is_instance_of_type(object_a, type_a):
    """Returns if `object_a` is an instance of `type_a`.

    Parameters
    ----------
    object_a: object
        The object to compare.
    type_a: type or typing._GenericAlias
        The subclass type.

    Returns
    -------
    bool
        True if `object_a` is an instance of `type_a`.
    """

    return is_type_subclass_of_type(type(object_a), type_a)
