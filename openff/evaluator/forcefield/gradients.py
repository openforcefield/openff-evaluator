import numpy
import pint.compat
from openff.units import unit


class ParameterGradientKey:
    @property
    def tag(self):
        return self._tag

    @property
    def smirks(self):
        return self._smirks

    @property
    def attribute(self):
        return self._attribute

    @property
    def virtual_site_type(self):
        return self._virtual_site_type

    @property
    def virtual_site_name(self):
        return self._virtual_site_name

    @property
    def virtual_site_match(self):
        return self._virtual_site_match

    def __init__(
        self,
        tag=None,
        smirks=None,
        attribute=None,
        virtual_site_type=None,
        virtual_site_name=None,
        virtual_site_match=None,
    ):
        self._tag = tag
        self._smirks = smirks
        self._attribute = attribute
        self._virtual_site_type = virtual_site_type
        self._virtual_site_name = virtual_site_name
        self._virtual_site_match = virtual_site_match

    def __getstate__(self):
        return {
            "tag": self._tag,
            "smirks": self._smirks,
            "attribute": self._attribute,
            "virtual_site_type": self._virtual_site_type,
            "virtual_site_name": self._virtual_site_name,
            "virtual_site_match": self._virtual_site_match,
        }

    def __setstate__(self, state):
        self._tag = state["tag"]
        self._smirks = state["smirks"]
        self._attribute = state["attribute"]
        # Keep deserialization tolerant of older payloads that predate
        # VirtualSite identity metadata.
        self._virtual_site_type = state.get("virtual_site_type")
        self._virtual_site_name = state.get("virtual_site_name")
        self._virtual_site_match = state.get("virtual_site_match")

    def __str__(self):
        return (
            f"tag={self._tag} smirks={self._smirks} attribute={self._attribute} "
            f"virtual_site_type={self._virtual_site_type} "
            f"virtual_site_name={self._virtual_site_name} "
            f"virtual_site_match={self._virtual_site_match}"
        )

    def __repr__(self):
        return f"<ParameterGradientKey {str(self)}>"

    def __hash__(self):
        return hash(
            (
                self._tag,
                self._smirks,
                self._attribute,
                self._virtual_site_type,
                self._virtual_site_name,
                self._virtual_site_match,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, ParameterGradientKey)
            and self._tag == other._tag
            and self._smirks == other._smirks
            and self._attribute == other._attribute
            and self._virtual_site_type == other._virtual_site_type
            and self._virtual_site_name == other._virtual_site_name
            and self._virtual_site_match == other._virtual_site_match
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class ParameterGradient:
    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    def __init__(self, key=None, value=None):
        self._key = key
        self._value = value

    def __getstate__(self):
        return {
            "key": self._key,
            "value": self._value,
        }

    def __setstate__(self, state):
        self._key = state["key"]
        self._value = state["value"]

    def __str__(self):
        return f"key=({self._key}) value={self._value}"

    def __repr__(self):
        return f"<ParameterGradient key={self._key} value={self._value}>"

    def __add__(self, other):
        """
        Parameters
        ----------
        other: ParameterGradient
        """
        if not isinstance(other, ParameterGradient):
            raise ValueError("Only ParameterGradient objects can be added together.")

        elif other.key != self.key:
            raise ValueError(
                "Only ParameterGradient objects with the same key can be added together."
            )

        return ParameterGradient(self.key, self.value + other.value)

    def __sub__(self, other):
        """
        Parameters
        ----------
        other: ParameterGradient
        """
        if not isinstance(other, ParameterGradient):
            raise ValueError("Only ParameterGradient objects can be subtracted.")

        elif other.key != self.key:
            raise ValueError(
                "Only ParameterGradient objects with the same key can be subtracted."
            )

        return ParameterGradient(self.key, self.value - other.value)

    def __mul__(self, other):
        """
        Parameters
        ----------
        other: float, int, openff.evaluator.unit.Quantity
        """

        if (
            not isinstance(other, float)
            and not isinstance(other, int)
            and not isinstance(other, unit.Quantity)
        ):
            raise ValueError(
                "ParameterGradient objects can only be multiplied by int's, "
                "float's or Quantity objects."
            )

        return ParameterGradient(self.key, self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Parameters
        ----------
        other: float, int, openff.evaluator.unit.Quantity
        """

        if (
            not isinstance(other, float)
            and not isinstance(other, int)
            and not isinstance(other, unit.Quantity)
        ):
            raise ValueError(
                "ParameterGradient objects can only be divided by int's, "
                "float's or Quantity objects."
            )

        return ParameterGradient(self.key, self.value / other)

    def __eq__(self, other):
        return (
            isinstance(other, ParameterGradient)
            and self.key == other.key
            and numpy.allclose(self.value, other.value)
        )


pint.compat.upcast_type_map[
    "openff.evaluator.forcefield.gradients.ParameterGradient"
] = ParameterGradient
