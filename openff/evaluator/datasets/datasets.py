"""
An API for defining, storing, and loading sets of physical
property data.
"""

import abc
import hashlib
import json
import re
import sys
import uuid
from enum import IntFlag, unique

import numpy
import pandas
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from openff.evaluator.datasets import CalculationSource, MeasurementSource, Source
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.serialization import TypedBaseModel, TypedJSONEncoder


@unique
class PropertyPhase(IntFlag):
    """An enum describing the phase that a property was
    collected in.

    Examples
    --------
    Properties measured in multiple phases (e.g. enthalpies of
    vaporization) can be defined be concatenating `PropertyPhase`
    enums:

    >>> gas_liquid_phase = PropertyPhase.Gas | PropertyPhase.Liquid
    """

    Undefined = 0x00
    Solid = 0x01
    Liquid = 0x02
    Gas = 0x04

    @classmethod
    def from_string(cls, enum_string):
        """Parses a phase enum from its string representation.

        Parameters
        ----------
        enum_string: str
            The str representation of a `PropertyPhase`

        Returns
        -------
        PropertyPhase
            The created enum

        Examples
        --------
        To round-trip convert a phase enum:
        >>> phase = PropertyPhase.Liquid | PropertyPhase.Gas
        >>> phase_str = str(phase)
        >>> parsed_phase = PropertyPhase.from_string(phase_str)
        """

        if len(enum_string) == 0:
            return PropertyPhase.Undefined

        components = [cls[x] for x in enum_string.split(" + ")]

        if len(components) == 0:
            return PropertyPhase.Undefined

        enum_value = components[0]

        for component in components[1:]:
            enum_value |= component

        return enum_value

    def __str__(self):
        return " + ".join([phase.name for phase in PropertyPhase if self & phase])

    def __repr__(self):
        return f"<PropertyPhase {str(self)}>"


class PhysicalProperty(AttributeClass, abc.ABC):
    """Represents the value of any physical property and it's uncertainty
    if provided.

    It additionally stores the thermodynamic state at which the property
    was collected, the phase it was collected in, information about
    the composition of the observed system, and metadata about how the
    property was collected.
    """

    @classmethod
    @abc.abstractmethod
    def default_unit(cls):
        """openff.evaluator.unit.Unit: The default unit (e.g. g / mol) associated with this
        class of property."""
        raise NotImplementedError()

    id = Attribute(
        docstring="A unique identifier string assigned to this property",
        type_hint=str,
        default_value=lambda: str(uuid.uuid4()).replace("-", ""),
    )

    substance = Attribute(
        docstring="The substance that this property was measured estimated for.",
        type_hint=Substance,
    )
    phase = Attribute(
        docstring="The phase / phases that this property was measured in.",
        type_hint=PropertyPhase,
    )
    thermodynamic_state = Attribute(
        docstring="The thermodynamic state that this property"
        "was measured / estimated at.",
        type_hint=ThermodynamicState,
    )

    value = Attribute(
        docstring="The measured / estimated value of this property.",
        type_hint=unit.Quantity,
    )
    uncertainty = Attribute(
        docstring="The uncertainty in measured / estimated value of this property.",
        type_hint=unit.Quantity,
        optional=True,
    )

    source = Attribute(
        docstring="The original source of this physical property.",
        type_hint=Source,
        optional=True,
    )
    metadata = Attribute(
        docstring="Additional metadata associated with this property. All property "
        "metadata will be made accessible to estimation workflows.",
        type_hint=dict,
        optional=True,
    )

    gradients = Attribute(
        docstring="The gradients of this property with respect to "
        "different force field parameters.",
        type_hint=list,
        optional=True,
    )

    def __init__(
        self,
        thermodynamic_state=None,
        phase=PropertyPhase.Undefined,
        substance=None,
        value=None,
        uncertainty=None,
        source=None,
    ):
        """Constructs a new PhysicalProperty object.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state that the property was measured in.
        phase : PropertyPhase
            The phase that the property was measured in.
        substance : Substance
            The composition of the substance that was measured.
        value: openff.evaluator.unit.Quantity
            The value of the measured physical property.
        uncertainty: openff.evaluator.unit.Quantity
            The uncertainty in the measured value.
        source: Source
            The source of this property.
        """
        if thermodynamic_state is not None:
            self.thermodynamic_state = thermodynamic_state
        if phase is not None:
            self.phase = phase

        if substance is not None:
            self.substance = substance

        if value is not None:
            self.value = value
        if uncertainty is not None:
            self.uncertainty = uncertainty

        self.gradients = []

        if source is not None:
            self.source = source

    def __setstate__(self, state):
        if "id" not in state:
            state["id"] = str(uuid.uuid4()).replace("-", "")

        super(PhysicalProperty, self).__setstate__(state)

    def _get_raw_property_hash(self) -> int:
        """
        Get the raw hash of a property based on its attributes.

        This method serializes the property attributes into a JSON string,
        sorts the keys, and computes a SHA-256 hash of the resulting string.
        The hash is then converted to an integer.

        Note: unlike `_get_property_hash`, this method does not truncate the hash value,
        so the number can be quite large.
        """

        type_ = type(self)
        clsname = f"{type_.__module__}.{type_.__qualname__}"

        obj = {
            "type": clsname,
            "substance": self.substance,
            "phase": self.phase,
            "thermodynamic_state": self.thermodynamic_state,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "source": self.source,
            "metadata": self.metadata,
        }
        serialized = json.dumps(obj, sort_keys=True, cls=TypedJSONEncoder)
        return int(hashlib.sha256(serialized.encode("utf-8")).hexdigest(), 16)
    
    def get_property_hash(self) -> int:
        """
        Returns a hash of the property based on attributes that are expected to
        have a meaningful value for the property. Hashes will change based on:

        - the property type
        - the value and uncertainty of the property
        - thermodynamic state
        - phase
        - substance
        - source
        - metadata

        The hash value will not depend on:
        - the id of the property (which is expected to be unique)
        - the gradients

        Note: as with the default __hash__ method in Python,
        the hash is truncated to the size of a Py_ssize_t, which is
        platform-dependent.

        Returns
        -------
        int
            The hash value of the property.
        """

        # hash() truncates the value returned from an object’s custom __hash__()
        # method to the size of a Py_ssize_t.

        # see https://docs.python.org/3/library/functions.html#hash
        # and https://github.com/python/cpython/blob/main/Include/cpython/pyhash.h#L8-L17

        raw_property_hash = self._get_raw_property_hash()

        # here we mimic the Python hash function for ease of comparison
        # and uses Mersenne primes for truncation
        if sys.hash_info.width == 64:
            mod = (1 << 61) - 1  # Mersenne prime for 64-bit hash
        else:
            mod = (1 << 31) - 1

        return raw_property_hash % mod

    def validate(self, attribute_type=None):
        super(PhysicalProperty, self).validate(attribute_type)

        assert self.value.units.dimensionality == self.default_unit().dimensionality

        if self.uncertainty != UNDEFINED:
            assert (
                self.uncertainty.units.dimensionality
                == self.default_unit().dimensionality
            )


class PhysicalPropertyDataSet(TypedBaseModel):
    """
    An object for storing and curating data sets of both physical property
    measurements and estimated. This class defines a number of convenience
    functions for filtering out unwanted properties, and for generating
    general statistics (such as the number of properties per substance)
    about the set.
    """

    def __init__(self):
        """
        Constructs a new PhysicalPropertyDataSet object.
        """
        self._properties = []

    @property
    def properties(self):
        """tuple of PhysicalProperty: A list of all of the properties
        within this set.
        """
        return tuple(self._properties)

    @property
    def property_types(self):
        """set of str: The types of property within this data set."""
        return set([x.__class__.__name__ for x in self._properties])

    @property
    def substances(self):
        """set of Substance: The substances for which the properties in this data set
        were collected for."""
        return set([x.substance for x in self._properties])

    @property
    def sources(self):
        """set of Source: The sources from which the properties in this data set were
        gathered."""
        return set([x.source for x in self._properties])

    def merge(self, data_set, validate=True):
        """Merge another data set into the current one.

        Parameters
        ----------
        data_set : PhysicalPropertyDataSet
            The secondary data set to merge into this one.
        validate: bool
            Whether to validate the other data set before merging.
        """
        if data_set is None:
            return

        self.add_properties(*data_set, validate=validate)

    def add_properties(self, *physical_properties, validate=True):
        """Adds a physical property to the data set.

        Parameters
        ----------
        physical_properties: PhysicalProperty
            The physical property to add.
        validate: bool
            Whether to validate the properties before adding them
            to the set.
        """

        all_ids = set(x.id for x in self)

        # TODO: Do we need to check for adding the same property twice?
        for physical_property in physical_properties:
            if validate:
                physical_property.validate()

            if physical_property.id in all_ids:
                raise KeyError(
                    f"A property with the unique id {physical_property.id} already "
                    f"exists."
                )

            all_ids.add(physical_property.id)

        self._properties.extend(physical_properties)

    def properties_by_substance(self, substance):
        """A generator which may be used to loop over all of the properties
        which were measured for a particular substance.

        Parameters
        ----------
        substance: Substance
            The substance of interest.

        Returns
        -------
        generator of PhysicalProperty
        """

        for physical_property in self._properties:
            if physical_property.substance != substance:
                continue

            yield physical_property

    def properties_by_type(self, property_type):
        """A generator which may be used to loop over all of properties
        of a particular type, e.g. all "Density" properties.

        Parameters
        ----------
        property_type: str or type of PhysicalProperty
            The type of property of interest. This may either be the string
            class name of the property or the class type.

        Returns
        -------
        generator of PhysicalProperty
        """

        if not isinstance(property_type, str):
            property_type = property_type.__name__

        for physical_property in self._properties:
            if physical_property.__class__.__name__ != property_type:
                continue

            yield physical_property

    def validate(self):
        """Checks to ensure that all properties within
        the set are valid physical property object.
        """
        for physical_property in self._properties:
            physical_property.validate()

    def to_pandas(self):
        """Converts a `PhysicalPropertyDataSet` to a `pandas.DataFrame` object
        with columns of

            - 'Id'
            - 'Temperature (K)'
            - 'Pressure (kPa)'
            - 'Phase'
            - 'N Components'
            - 'Component 1'
            - 'Role 1'
            - 'Mole Fraction 1'
            - 'Exact Amount 1'
            - ...
            - 'Component N'
            - 'Role N'
            - 'Mole Fraction N'
            - 'Exact Amount N'
            - '<Property 1> Value (<default unit>)'
            - '<Property 1> Uncertainty / (<default unit>)'
            - ...
            - '<Property N> Value / (<default unit>)'
            - '<Property N> Uncertainty / (<default unit>)'
            - `'Source'`

        where 'Component X' is a column containing the smiles representation of
        component X.

        Returns
        -------
        pandas.DataFrame
            The create data frame.
        """

        if len(self) == 0:
            return pandas.DataFrame()

        # Keep track of the maximum number of components in any substance
        # as this determines the number of component columns.
        maximum_number_of_components = 0

        data_rows = []

        # Extract the data from the data set.
        default_units = {}

        for physical_property in self:
            # Extract the measured state.
            temperature = physical_property.thermodynamic_state.temperature.to(
                unit.kelvin
            ).magnitude
            pressure = None

            if physical_property.thermodynamic_state.pressure != UNDEFINED:
                pressure = physical_property.thermodynamic_state.pressure.to(
                    unit.kilopascal
                ).magnitude

            phase = str(physical_property.phase)

            # Extract the component data.
            components = []
            amounts = []
            roles = []

            for index, component in enumerate(physical_property.substance):
                component_amounts = {MoleFraction: None, ExactAmount: None}

                for x in physical_property.substance.get_amounts(component):
                    assert isinstance(x, (MoleFraction, ExactAmount))
                    component_amounts[type(x)] = x.value

                components.append(component.smiles)
                amounts.append(component_amounts)
                roles.append(component.role.name)

            # Extract the value data as a string.
            default_unit = physical_property.default_unit()
            default_units[physical_property.__class__.__name__] = default_unit

            value = (
                None
                if physical_property.value == UNDEFINED
                else physical_property.value.to(default_unit).magnitude
            )
            uncertainty = (
                None
                if physical_property.uncertainty == UNDEFINED
                else physical_property.uncertainty.to(default_unit).magnitude
            )

            # Extract the data source.
            source = None

            if isinstance(physical_property.source, MeasurementSource):
                source = physical_property.source.doi

                if source is None or len(source) == 0:
                    source = physical_property.source.reference

            elif isinstance(physical_property.source, CalculationSource):
                source = physical_property.source.fidelity

            # Create the data row.
            data_row = {
                "Id": physical_property.id,
                "Temperature (K)": temperature,
                "Pressure (kPa)": pressure,
                "Phase": phase,
                "N Components": len(physical_property.substance),
            }

            for index in range(len(components)):
                data_row[f"Component {index + 1}"] = components[index]
                data_row[f"Role {index + 1}"] = roles[index]
                data_row[f"Mole Fraction {index + 1}"] = amounts[index][MoleFraction]
                data_row[f"Exact Amount {index + 1}"] = amounts[index][ExactAmount]

            data_row[f"{type(physical_property).__name__} Value ({default_unit:~})"] = (
                value
            )
            data_row[
                f"{type(physical_property).__name__} Uncertainty ({default_unit:~})"
            ] = uncertainty

            data_row["Source"] = source

            data_rows.append(data_row)

            maximum_number_of_components = max(
                maximum_number_of_components, len(physical_property.substance)
            )

        # Set up the column headers.
        if len(data_rows) == 0:
            return None

        data_columns = [
            "Id",
            "Temperature (K)",
            "Pressure (kPa)",
            "Phase",
            "N Components",
        ]

        for index in range(maximum_number_of_components):
            data_columns.append(f"Component {index + 1}")
            data_columns.append(f"Role {index + 1}")
            data_columns.append(f"Mole Fraction {index + 1}")
            data_columns.append(f"Exact Amount {index + 1}")

        for property_type in self.property_types:
            default_unit = default_units[property_type]

            data_columns.append(f"{property_type} Value ({default_unit:~})")
            data_columns.append(f"{property_type} Uncertainty ({default_unit:~})")

        data_columns.append("Source")

        data_frame = pandas.DataFrame(data_rows, columns=data_columns)
        return data_frame

    @classmethod
    def from_pandas(cls, data_frame: pandas.DataFrame) -> "PhysicalPropertyDataSet":
        """Constructs a data set object from a pandas ``DataFrame`` object.

        Notes
        -----
        * All physical properties are assumed to be source from experimental
          measurements.
        * Currently this method onlu supports data frames containing properties
          which are built-in to the framework (e.g. Density).
        * This method assumes the data frame has a structure identical to that
          produced by the ``PhysicalPropertyDataSet.to_pandas`` function.

        Parameters
        ----------
        data_frame
            The data frame to construct the data set from.

        Returns
        -------
            The constructed data set.
        """

        from openff.evaluator import properties

        property_header_matches = {
            re.match(r"^([a-zA-Z]+) Value \(([a-zA-Z0-9+-/\s*^]*)\)$", header)
            for header in data_frame
            if header.find(" Value ") >= 0
        }
        property_headers = {}

        # Validate that the headers have the correct format, specify a
        # built-in property type, and specify correctly the properties
        # units.
        for match in property_header_matches:
            assert match

            property_type_string, property_unit_string = match.groups()

            assert hasattr(properties, property_type_string)
            property_type = getattr(properties, property_type_string)

            property_unit = unit.Unit(property_unit_string)
            assert property_unit is not None

            assert (
                property_unit.dimensionality
                == property_type.default_unit().dimensionality
            )

            property_headers[match.group(0)] = (property_type, property_unit)

        # Convert the data rows to property objects.
        physical_properties = []

        # Drop data point if thermophysical data is not included (see #578)
        data_frame = data_frame.dropna(
            subset=[
                "Pressure (kPa)",
                "Temperature (K)",
                "Phase",
            ]
        )

        for _, data_row in data_frame.iterrows():
            data_row = data_row.dropna()

            # Extract the state at which the measurement was made.
            thermodynamic_state = ThermodynamicState(
                temperature=data_row["Temperature (K)"] * unit.kelvin,
                pressure=data_row["Pressure (kPa)"] * unit.kilopascal,
            )
            property_phase = PropertyPhase.from_string(data_row["Phase"])

            # Extract the substance the measurement was made for.
            substance = Substance()

            for i in range(data_row["N Components"]):
                component = Component(
                    smiles=data_row[f"Component {i + 1}"],
                    role=Component.Role[data_row.get(f"Role {i + 1}", "Solvent")],
                )

                mole_fraction = data_row.get(f"Mole Fraction {i + 1}", 0.0)
                exact_amount = data_row.get(f"Exact Amount {i + 1}", 0)

                if not numpy.isclose(mole_fraction, 0.0):
                    substance.add_component(component, MoleFraction(mole_fraction))
                if not numpy.isclose(exact_amount, 0.0):
                    substance.add_component(component, ExactAmount(exact_amount))

            for (
                property_header,
                (property_type, property_unit),
            ) in property_headers.items():
                # Check to see whether the row contains a value for this
                # type of property.
                if property_header not in data_row:
                    continue

                uncertainty_header = property_header.replace("Value", "Uncertainty")

                source_string = data_row["Source"]

                is_doi = all(
                    any(
                        re.match(pattern, split_string, re.I)
                        for pattern in [
                            r"^10.\d{4,9}/[-._;()/:A-Z0-9]+$",
                            r"^10.1002/[^\s]+$",
                            r"^10.\d{4}/\d+-\d+X?(\d+)\d+<[\d\w]+:[\d\w]*>\d+.\d+.\w+;\d$",
                            r"^10.1021/\w\w\d+$",
                            r"^10.1207/[\w\d]+\&\d+_\d+$",
                        ]
                    )
                    for split_string in source_string.split(" + ")
                )

                physical_property = property_type(
                    thermodynamic_state=thermodynamic_state,
                    phase=property_phase,
                    value=data_row[property_header] * property_unit,
                    uncertainty=(
                        None
                        if uncertainty_header not in data_row
                        else data_row[uncertainty_header] * property_unit
                    ),
                    substance=substance,
                    source=MeasurementSource(
                        doi="" if not is_doi else source_string,
                        reference=source_string if not is_doi else "",
                    ),
                )

                identifier = data_row.get("Id", None)

                if identifier:
                    physical_property.id = identifier

                physical_properties.append(physical_property)

        data_set = PhysicalPropertyDataSet()
        data_set.add_properties(*physical_properties)

        return data_set

    def __len__(self):
        return len(self._properties)

    def __iter__(self):
        return iter(self._properties)

    def __getstate__(self):
        return {"properties": self._properties}

    def __setstate__(self, state):
        self._properties = state["properties"]

        assert all(isinstance(x, PhysicalProperty) for x in self)

        # Ensure each property has a unique id.
        all_ids = set(x.id for x in self)
        assert len(all_ids) == len(self)

    def __str__(self):
        return (
            f"n_properties={len(self)} n_substances={len(self.substances)} "
            f"n_sources={len(self.sources)}"
        )

    def __repr__(self):
        return f"<PhysicalPropertyDataSet {str(self)}>"
