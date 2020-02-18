"""
An API for defining, storing, and loading sets of physical
property data.
"""
import uuid
from enum import IntFlag, unique

import pandas
import pint

from evaluator import unit
from evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from evaluator.datasets import CalculationSource, MeasurementSource, Source
from evaluator.substances import ExactAmount, MoleFraction, Substance
from evaluator.thermodynamics import ThermodynamicState
from evaluator.utils.serialization import TypedBaseModel


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


class PhysicalProperty(AttributeClass):
    """Represents the value of any physical property and it's uncertainty
    if provided.

    It additionally stores the thermodynamic state at which the property
    was collected, the phase it was collected in, information about
    the composition of the observed system, and metadata about how the
    property was collected.
    """

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
        type_hint=pint.Quantity,
    )
    uncertainty = Attribute(
        docstring="The uncertainty in measured / estimated value of this property.",
        type_hint=pint.Quantity,
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
        value: pint.Quantity
            The value of the measured physical property.
        uncertainty: pint.Quantity
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

    def merge(self, data_set):
        """Merge another data set into the current one.

        Parameters
        ----------
        data_set : PhysicalPropertyDataSet
            The secondary data set to merge into this one.
        """
        if data_set is None:
            return

        self.add_properties(*data_set)

    def add_properties(self, *physical_properties):
        """Adds a physical property to the data set.

        Parameters
        ----------
        physical_properties: PhysicalProperty
            The physical property to add.
        """

        all_ids = set(x.id for x in self)

        # TODO: Do we need to check for adding the same property twice?
        for physical_property in physical_properties:

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

    def filter_by_function(self, filter_function):
        """Filter the data set using a given filter function.

        Parameters
        ----------
        filter_function : lambda
            The filter function.
        """
        self._properties = list(filter(filter_function, self._properties))

    def filter_by_property_types(self, *property_types):
        """Filter the data set based on the type of property (e.g Density).

        Parameters
        ----------
        property_types : PropertyType or str
            The type of property which should be retained.

        Examples
        --------
        Filter the dataset to only contain densities and static dielectric constants

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from evaluator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> # Filter the dataset to only include densities and dielectric constants.
        >>> from evaluator.properties import Density, DielectricConstant
        >>> data_set.filter_by_property_types(Density, DielectricConstant)

        or

        >>> data_set.filter_by_property_types('Density', 'DielectricConstant')
        """

        property_types = [
            x if isinstance(x, str) else x.__name__ for x in property_types
        ]

        def filter_function(x):
            return x.__class__.__name__ in property_types

        self.filter_by_function(filter_function)

    def filter_by_phases(self, phases):
        """Filter the data set based on the phase of the property (e.g liquid).

        Parameters
        ----------
        phases : PropertyPhase
            The phase of property which should be retained.

        Examples
        --------
        Filter the dataset to only include liquid properties.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from evaluator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from evaluator.datasets import PropertyPhase
        >>> data_set.filter_by_temperature(PropertyPhase.Liquid)
        """

        def filter_function(x):
            return x.phase & phases

        self.filter_by_function(filter_function)

    def filter_by_temperature(self, min_temperature, max_temperature):
        """Filter the data set based on a minimum and maximum temperature.

        Parameters
        ----------
        min_temperature : pint.Quantity
            The minimum temperature.
        max_temperature : pint.Quantity
            The maximum temperature.

        Examples
        --------
        Filter the dataset to only include properties measured between 130-260 K.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from evaluator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from evaluator import unit
        >>> data_set.filter_by_temperature(min_temperature=130*unit.kelvin, max_temperature=260*unit.kelvin)
        """

        def filter_function(x):
            return (
                min_temperature <= x.thermodynamic_state.temperature <= max_temperature
            )

        self.filter_by_function(filter_function)

    def filter_by_pressure(self, min_pressure, max_pressure):
        """Filter the data set based on a minimum and maximum pressure.

        Parameters
        ----------
        min_pressure : pint.Quantity
            The minimum pressure.
        max_pressure : pint.Quantity
            The maximum pressure.

        Examples
        --------
        Filter the dataset to only include properties measured between 70-150 kPa.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from evaluator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from evaluator import unit
        >>> data_set.filter_by_temperature(min_pressure=70*unit.kilopascal, max_temperature=150*unit.kilopascal)
        """

        def filter_function(x):

            if x.thermodynamic_state.pressure == UNDEFINED:
                return True

            return min_pressure <= x.thermodynamic_state.pressure <= max_pressure

        self.filter_by_function(filter_function)

    def filter_by_components(self, number_of_components):
        """Filter the data set based on the number of components present
        in the substance the data points were collected for.

        Parameters
        ----------
        number_of_components : int
            The allowed number of components in the mixture.

        Examples
        --------
        Filter the dataset to only include pure substance properties.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from evaluator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> data_set.filter_by_components(number_of_components=1)
        """

        def filter_function(x):
            return x.substance.number_of_components == number_of_components

        self.filter_by_function(filter_function)

    def filter_by_elements(self, *allowed_elements):
        """Filters out those properties which were estimated for
         compounds which contain elements outside of those defined
         in `allowed_elements`.

        Parameters
        ----------
        allowed_elements: str
            The symbols (e.g. C, H, Cl) of the elements to
            retain.
        """
        from openforcefield.topology import Molecule

        def filter_function(physical_property):

            substance = physical_property.substance

            for component in substance.components:

                molecule = Molecule.from_smiles(
                    component.smiles, allow_undefined_stereo=True
                )

                if not all(
                    [x.element.symbol in allowed_elements for x in molecule.atoms]
                ):
                    return False

            return True

        self.filter_by_function(filter_function)

    def filter_by_smiles(self, *allowed_smiles):
        """Filters out those properties which were estimated for
         compounds which do not appear in the allowed `smiles` list.

        Parameters
        ----------
        allowed_smiles: str
            The smiles identifiers of the compounds to keep
            after filtering.
        """

        def filter_function(physical_property):

            substance = physical_property.substance

            for component in substance.components:

                if component.smiles in allowed_smiles:
                    continue

                return False

            return True

        self.filter_by_function(filter_function)

    def filter_by_uncertainties(self):
        """Filters out those properties which don't have
        their uncertainties reported.
        """

        def filter_function(physical_property):
            return physical_property.uncertainty is not None

        self.filter_by_function(filter_function)

    def to_pandas(self):
        """Converts a `PhysicalPropertyDataSet` to a `pandas.DataFrame` object
        with columns of

            - 'Temperature'
            - 'Pressure'
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
            - '<Property 1> Value'
            - '<Property 1> Uncertainty'
            - ...
            - '<Property N> Value'
            - '<Property N> Uncertainty'
            - `'Source'`

        where 'Component X' is a column containing the smiles representation of component X.

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
        for physical_property in self:

            # Extract the measured state.
            temperature = physical_property.thermodynamic_state.temperature.to(
                unit.kelvin
            )
            pressure = None

            if physical_property.thermodynamic_state.pressure != UNDEFINED:

                pressure = physical_property.thermodynamic_state.pressure.to(
                    unit.kilopascal
                )

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
            value = (
                None
                if physical_property.value == UNDEFINED
                else str(physical_property.value)
            )
            uncertainty = (
                None
                if physical_property.uncertainty == UNDEFINED
                else str(physical_property.uncertainty)
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
                "Temperature": str(temperature),
                "Pressure": str(pressure),
                "Phase": phase,
                "N Components": len(physical_property.substance),
            }

            for index in range(len(components)):

                data_row[f"Component {index + 1}"] = components[index]
                data_row[f"Role {index + 1}"] = roles[index]
                data_row[f"Mole Fraction {index + 1}"] = amounts[index][MoleFraction]
                data_row[f"Exact Amount {index + 1}"] = amounts[index][ExactAmount]

            data_row[f"{type(physical_property).__name__} Value"] = value
            data_row[f"{type(physical_property).__name__} Uncertainty"] = uncertainty

            data_row["Source"] = source

            data_rows.append(data_row)

            maximum_number_of_components = max(
                maximum_number_of_components, len(physical_property.substance)
            )

        # Set up the column headers.
        if len(data_rows) == 0:
            return None

        data_columns = [
            "Temperature",
            "Pressure",
            "Phase",
            "N Components",
        ]

        for index in range(maximum_number_of_components):
            data_columns.append(f"Component {index + 1}")
            data_columns.append(f"Role {index + 1}")
            data_columns.append(f"Mole Fraction {index + 1}")
            data_columns.append(f"Exact Amount {index + 1}")

        for property_type in self.property_types:
            data_columns.append(f"{property_type} Value")
            data_columns.append(f"{property_type} Uncertainty")

        data_columns.append("Source")

        data_frame = pandas.DataFrame(data_rows, columns=data_columns)
        return data_frame

    def __len__(self):
        return len(self._properties)

    def __iter__(self):
        return iter(self._properties)

    def __getstate__(self):
        return {"properties": self._properties}

    def __setstate__(self, state):

        self._properties = state["properties"]

        assert all(isinstance(x, PhysicalProperty) for x in self)

        for physical_property in self:
            physical_property.validate()

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
