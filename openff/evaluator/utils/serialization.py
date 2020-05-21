"""
A collection of classes which aid in serializing data types.
"""

import importlib
import inspect
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import dateutil.parser
import numpy as np
import pint

from openff.evaluator import unit


def _type_string_to_object(type_string):

    if type_string.startswith("evaluator."):
        # Make files produced with the beta evaluator release compatible with
        # the full evaluator release.
        type_string = type_string.replace("evaluator.", "openff.evaluator.")

    if type_string == "openff.evaluator.unit.Unit":
        return unit.Unit
    if type_string == "openff.evaluator.unit.Quantity":
        return unit.Quantity
    if type_string == "openff.evaluator.unit.Measurement":
        return unit.Measurement

    last_period_index = type_string.rfind(".")

    if last_period_index < 0 or last_period_index == len(type_string) - 1:
        raise ValueError(
            "The type string is invalid - it should be of the form "
            "module_path.class_name: {}".format(type_string)
        )

    type_string_split = type_string.split(".")

    class_object = None
    module_path = None

    while len(type_string_split) > 0:

        class_name = type_string_split.pop(0)

        try:

            if module_path is None:
                module_path = class_name
            else:
                module_path = module_path + "." + class_name

            # First try and treat the current string as a module
            module = importlib.import_module(module_path)
            class_object = module

        except ImportError:

            # If we get an import error, try then to treat the string
            # as the name of a nested class.
            class_object = getattr(class_object, class_name)

    return class_object


def _type_to_type_string(object_type):
    """Converts a type to a serializable string.

    Parameters
    ----------
    object_type: type
        The type to convert.

    Returns
    -------
    str
        The converted type.
    """

    if (
        issubclass(object_type, pint.Unit)
        or f"{object_type.__module__}.{object_type.__qualname__}"
        == "pint.quantity.build_quantity_class.<locals>.Unit"
    ):
        return "openff.evaluator.unit.Unit"

    if (
        issubclass(object_type, pint.Measurement)
        or f"{object_type.__module__}.{object_type.__qualname__}"
        == "pint.quantity.build_quantity_class.<locals>.Measurement"
    ):

        return "openff.evaluator.unit.Measurement"

    if (
        issubclass(object_type, pint.Quantity)
        or f"{object_type.__module__}.{object_type.__qualname__}"
        == "pint.quantity.build_quantity_class.<locals>.Quantity"
    ):
        return "openff.evaluator.unit.Quantity"

    qualified_name = object_type.__qualname__

    return_value = "{}.{}".format(object_type.__module__, qualified_name)

    if return_value.startswith("evaluator."):
        return_value = return_value.replace("evaluator.", "openff.evaluator.")

    return return_value


def serialize_quantity(quantity):
    """Serializes a pint.Quantity into a dictionary of the form
    `{'value': quantity.value_in_unit(quantity.unit), 'unit': quantity.unit}`

    Parameters
    ----------
    quantity : pint.Quantity
        The quantity to serialize

    Returns
    -------
    dict of str and str
        A dictionary representation of a pint.Quantity
        with keys of {"value", "unit"}
    """

    value = quantity.magnitude
    return {"value": value, "unit": str(quantity.units)}


def deserialize_quantity(serialized):
    """Deserialize a pint.Quantity from a dictionary.

    Parameters
    ----------
    serialized : dict of str and str
        A dictionary representation of a pint.Quantity
        which must have keys {"value", "unit"}

    Returns
    -------
    pint.Quantity
        The deserialized quantity.
    """

    if "@type" in serialized:
        serialized.pop("@type")

    value_unit = unit.dimensionless

    if serialized["unit"] is not None:
        value_unit = unit(serialized["unit"])

    return serialized["value"] * value_unit


def serialize_measurement(measurement):
    """Serializes a `pint.Measurement` into a dictionary of the form
    `{'value', 'error'}`.

    Parameters
    ----------
    measurement : pint.Measurement
        The measurement to serialize

    Returns
    -------
    dict of str and str
        A dictionary representation of a pint.Measurement
        with keys of {"value", "error"}
    """
    return {"value": measurement.value, "error": measurement.error}


def deserialize_measurement(serialized):
    """Deserialize a `pint.Measurement` from a dictionary of the form
    `{'value', 'error'}`.

    Parameters
    ----------
    serialized : dict of str and str
        A dictionary representation of a `pint.Measurement`
        which must have keys {"value", "error"}

    Returns
    -------
    pint.Measurement
        The deserialized measurement.
    """

    if "@type" in serialized:
        serialized.pop("@type")

    return serialized["value"].plus_minus(serialized["error"])


def serialize_enum(enum):

    if not isinstance(enum, Enum):
        raise ValueError("{} is not an Enum".format(type(enum)))

    return {"value": enum.value}


def deserialize_enum(enum_dictionary):

    if "@type" not in enum_dictionary:

        raise ValueError(
            "The serialized enum dictionary must includewhich type the enum is."
        )

    if "value" not in enum_dictionary:

        raise ValueError("The serialized enum dictionary must includethe enum value.")

    enum_type_string = enum_dictionary["@type"]
    enum_value = enum_dictionary["value"]

    enum_class = _type_string_to_object(enum_type_string)

    if not issubclass(enum_class, Enum):
        raise ValueError("<{}> is not an Enum".format(enum_class))

    return enum_class(enum_value)


def serialize_set(set_object):

    if not isinstance(set_object, set):
        raise ValueError("{} is not a set".format(type(set)))

    return {"value": list(set_object)}


def deserialize_set(set_dictionary):

    if "value" not in set_dictionary:

        raise ValueError(
            "The serialized set dictionary must includethe value of the set."
        )

    set_value = set_dictionary["value"]

    if not isinstance(set_value, list):

        raise ValueError("The value of the serialized set must be a list.")

    return set(set_value)


def serialize_frozen_set(set_object):

    if not isinstance(set_object, frozenset):
        raise ValueError("{} is not a frozenset".format(type(frozenset)))

    return {"value": list(set_object)}


def deserialize_frozen_set(set_dictionary):

    if "value" not in set_dictionary:

        raise ValueError(
            "The serialized frozenset dictionary must includethe value of the set."
        )

    set_value = set_dictionary["value"]

    if not isinstance(set_value, list):
        raise ValueError("The value of the serialized set must be a list.")

    return frozenset(set_value)


class TypedJSONEncoder(json.JSONEncoder):

    _natively_supported_types = [dict, list, tuple, str, int, float, bool]

    _custom_supported_types = {
        Enum: serialize_enum,
        pint.Measurement: serialize_measurement,
        pint.Quantity: serialize_quantity,
        set: serialize_set,
        frozenset: serialize_frozen_set,
        np.float16: lambda x: {"value": float(x)},
        np.float32: lambda x: {"value": float(x)},
        np.float64: lambda x: {"value": float(x)},
        np.int32: lambda x: {"value": int(x)},
        np.int64: lambda x: {"value": int(x)},
        np.ndarray: lambda x: {"value": x.tolist()},
        datetime: lambda x: {"value": x.isoformat()},
    }

    def default(self, value_to_serialize):

        if value_to_serialize is None:
            return None

        type_to_serialize = type(value_to_serialize)

        if type_to_serialize in TypedJSONEncoder._natively_supported_types:
            # If the value is a native type, then let the default serializer
            # handle it.
            return super(TypedJSONEncoder, self).default(value_to_serialize)

        # Otherwise, we need to add a @type attribute to it.
        type_tag = _type_to_type_string(type_to_serialize)

        if type_tag == "openff.evaluator.unit.Unit":
            type_to_serialize = unit.Unit
        if type_tag == "openff.evaluator.unit.Quantity":
            type_to_serialize = pint.Quantity
        if type_tag == "openff.evaluator.unit.Measurement":
            type_to_serialize = pint.Measurement

        custom_encoder = None

        for encoder_type in TypedJSONEncoder._custom_supported_types:

            if isinstance(encoder_type, str):

                qualified_name = type_to_serialize.__qualname__

                if encoder_type != qualified_name:
                    continue

            elif not issubclass(type_to_serialize, encoder_type):
                continue

            custom_encoder = TypedJSONEncoder._custom_supported_types[encoder_type]
            break

        if custom_encoder is not None:

            try:
                serializable_dictionary = custom_encoder(value_to_serialize)

            except Exception as e:

                raise ValueError(
                    "{} ({}) could not be serialized "
                    "using a specialized custom encoder: {}".format(
                        value_to_serialize, type_to_serialize, e
                    )
                )

        elif hasattr(value_to_serialize, "__getstate__"):

            try:
                serializable_dictionary = value_to_serialize.__getstate__()

            except Exception as e:

                raise ValueError(
                    "{} ({}) could not be serialized "
                    "using its __getstate__ method: {}".format(
                        value_to_serialize, type_to_serialize, e
                    )
                )

        else:

            raise ValueError(
                "Objects of type {} are not serializable, please either"
                "add a __getstate__ method, or add the object to the list"
                "of custom supported types.".format(type_to_serialize)
            )

        serializable_dictionary["@type"] = type_tag
        return serializable_dictionary


class TypedJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    _custom_supported_types = {
        Enum: deserialize_enum,
        pint.Measurement: deserialize_measurement,
        pint.Quantity: deserialize_quantity,
        set: deserialize_set,
        frozenset: deserialize_frozen_set,
        np.float16: lambda x: np.float16(x["value"]),
        np.float32: lambda x: np.float32(x["value"]),
        np.float64: lambda x: np.float64(x["value"]),
        np.int32: lambda x: np.int32(x["value"]),
        np.int64: lambda x: np.int64(x["value"]),
        np.ndarray: lambda x: np.array(x["value"]),
        datetime: lambda x: dateutil.parser.parse(x["value"]),
    }

    @staticmethod
    def object_hook(object_dictionary):

        if "@type" not in object_dictionary:
            return object_dictionary

        type_string = object_dictionary["@type"]
        class_type = _type_string_to_object(type_string)

        custom_decoder = None

        for decoder_type in TypedJSONDecoder._custom_supported_types:

            if isinstance(decoder_type, str):

                if decoder_type != class_type.__qualname__:
                    continue

            elif not issubclass(class_type, decoder_type):
                continue

            custom_decoder = TypedJSONDecoder._custom_supported_types[decoder_type]
            break

        if custom_decoder is not None:

            try:
                deserialized_object = custom_decoder(object_dictionary)

            except Exception as e:

                raise ValueError(
                    "{} ({}) could not be deserialized "
                    "using a specialized custom decoder: {}".format(
                        object_dictionary, type(class_type), e
                    )
                )

        elif hasattr(class_type, "__setstate__"):

            class_init_signature = inspect.signature(class_type)

            for parameter in class_init_signature.parameters.values():

                if (
                    parameter.default != inspect.Parameter.empty
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    or parameter.kind == inspect.Parameter.VAR_POSITIONAL
                ):

                    continue

                raise ValueError(
                    f"Cannot deserialize objects ({class_type}) which have non-"
                    f"optional arguments {parameter.name} in the constructor."
                )

            deserialized_object = class_type()
            deserialized_object.__setstate__(object_dictionary)

        else:

            raise ValueError(
                "Objects of type {} are not deserializable, please either"
                "add a __setstate__ method, or add the object to the list"
                "of custom supported types.".format(type(class_type))
            )

        return deserialized_object


class TypedBaseModel(ABC):
    """An abstract base class which represents any object which
    can be serialized to JSON.

    JSON produced using this class will include extra @type tags
    for any non-primitive typed values (e.g not a str, int...),
    which ensure that the correct class structure is correctly
    reproduced on deserialization.

    EXAMPLE

    It is a requirement that any classes inheriting from this one
    must implement a valid `__getstate__` and `__setstate__` method,
    as these are what determines the structure of the serialized
    output.
    """

    def json(self, file_path=None, format=False):
        """Creates a JSON representation of this class.

        Parameters
        ----------
        file_path: str, optional
            The (optional) file path to save the JSON file to.
        format: bool
            Whether to format the JSON or not.

        Returns
        -------
        str
            The JSON representation of this class.
        """
        if format:
            json_string = json.dumps(
                self,
                sort_keys=True,
                indent=2,
                separators=(",", ": "),
                cls=TypedJSONEncoder,
            )

        else:
            json_string = json.dumps(self, cls=TypedJSONEncoder)

        if file_path is not None:

            with open(file_path, "w") as file:
                file.write(json_string)

        return json_string

    @classmethod
    def from_json(cls, file_path):
        """Create this object from a JSON file.

        Parameters
        ----------
        file_path: str
            The path to load the JSON from.

        Returns
        -------
        cls
            The parsed class.
        """
        with open(file_path, "r") as file:
            return cls.parse_json(file.read())

    @classmethod
    def parse_json(cls, string_contents, encoding="utf8"):
        """Parses a typed json string into the corresponding class
        structure.

        Parameters
        ----------
        string_contents: str or bytes
            The typed json string.
        encoding: str
            The encoding of the `string_contents`.

        Returns
        -------
        Any
            The parsed class.
        """
        return_object = json.loads(
            string_contents, encoding=encoding, cls=TypedJSONDecoder
        )
        return return_object

    @abstractmethod
    def __getstate__(self):
        """Returns a dictionary representation of this object.

        Returns
        -------
        dict of str, Any
            The dictionary representation of this object.
        """
        pass

    @abstractmethod
    def __setstate__(self, state):
        """Sets the fields of this object from its dictionary representation.

        Parameters
        ----------
        state: dict of str, Any
            The dictionary representation of the object.
        """
        pass
