"""
An API for defining and creating substances.
"""

import abc
import math
import typing

import numpy as np

from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass


class Amount(AttributeClass, abc.ABC):
    """A representation of the amount of a given component
    in a `Substance`.
    """

    value = Attribute(
        docstring="The value of this amount.",
        type_hint=typing.Union[float, int],
        read_only=True,
    )

    def __init__(self, value=UNDEFINED):
        """
        Parameters
        ----------
        value: float or int
            The value of this amount.
        """
        self._set_value("value", value)

    @property
    def identifier(self):
        """A string identifier for this amount."""
        raise NotImplementedError()

    @abc.abstractmethod
    def to_number_of_molecules(self, total_substance_molecules, tolerance=None):
        """Converts this amount to an exact number of molecules

        Parameters
        ----------
        total_substance_molecules: int
            The total number of molecules in the whole substance. This amount
            will contribute to a portion of this total number.
        tolerance: float, optional
            The tolerance with which this amount should be in. As an example,
            when converting a mole fraction into a number of molecules, the
            total number of molecules may not be sufficiently large enough to
            reproduce this amount.

        Returns
        -------
        int
            The number of molecules which this amount represents,
            given the `total_substance_molecules`.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return f"<{self.__class__.__name__} {str(self)}>"

    def __eq__(self, other):
        return type(self) == type(other) and np.isclose(self.value, other.value)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.identifier)


class MoleFraction(Amount):
    """The mole fraction of a `Component` in a `Substance`."""

    value = Attribute(docstring="The value of this amount.", type_hint=float)

    @property
    def identifier(self):
        return f"x={self.value:.6f}"

    def to_number_of_molecules(self, total_substance_molecules, tolerance=None):

        # Determine how many molecules of each type will be present in the system.
        number_of_molecules = self.value * total_substance_molecules
        fractional_number_of_molecules = number_of_molecules % 1

        if np.isclose(fractional_number_of_molecules, 0.5):
            number_of_molecules = int(number_of_molecules)
        else:
            number_of_molecules = int(round(number_of_molecules))

        if number_of_molecules == 0:

            raise ValueError(
                "The total number of substance molecules was not large enough, "
                "such that this non-zero amount translates into zero molecules "
                "of this component in the substance."
            )

        if tolerance is not None:

            mole_fraction = number_of_molecules / total_substance_molecules

            if abs(mole_fraction - self.value) > tolerance:

                raise ValueError(
                    f"The mole fraction ({mole_fraction}) given a total number of molecules "
                    f"({total_substance_molecules}) is outside of the tolerance {tolerance} "
                    f"of the target mole fraction {self.value}"
                )

        return number_of_molecules

    def validate(self, attribute_type=None):
        super(MoleFraction, self).validate(attribute_type)

        if self.value <= 0.0 or self.value > 1.0:

            raise ValueError(
                "A mole fraction must be greater than zero, and less than or "
                "equal to one."
            )

        if math.floor(self.value * 1e6) < 1:

            raise ValueError(
                "Mole fractions are only precise to the sixth "
                "decimal place within this class representation."
            )


class ExactAmount(Amount):
    """The exact number of instances of a `Component` in a `Substance`.

    An assumption is made that this amount is for a component which is
    infinitely dilute (such as ligands in binding calculations), and hence
    do not contribute to the total mole fraction of a `Substance`.
    """

    value = Attribute(docstring="The value of this amount.", type_hint=int)

    @property
    def identifier(self):
        return f"n={int(round(self.value)):d}"

    def to_number_of_molecules(self, total_substance_molecules, tolerance=None):
        return self.value
