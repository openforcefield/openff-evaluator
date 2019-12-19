"""
An API for defining, storing, and loading sets of physical
property data.
"""
import uuid
from collections import defaultdict
from enum import IntFlag, unique

import pandas
from simtk.openmm.app import element

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass
from propertyestimator.datasets import CalculationSource, MeasurementSource, Source
from propertyestimator.substances import MoleFraction, Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import create_molecule_from_smiles
from propertyestimator.utils.serialization import TypedBaseModel


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
        default_value=lambda: str(uuid.uuid4()),
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
        value: unit.Quantity
            The value of the measured physical property.
        uncertainty: unit.Quantity
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
            state["id"] = str(uuid.uuid4())

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
        self._properties = {}
        self._sources = []

    @property
    def properties(self):
        """
        dict of str and list of PhysicalProperty: A list of all of the properties
        within this set, partitioned by substance identifier.

        See Also
        --------
        Substance.identifier()
        """
        return self._properties

    @property
    def sources(self):
        """list of Source: The list of sources from which the properties were gathered"""
        return self._sources

    @property
    def number_of_properties(self):
        """int: The number of properties in the data set."""
        return sum([len(properties) for properties in self._properties.values()])

    def merge(self, data_set):
        """Merge another data set into the current one.

        Parameters
        ----------
        data_set : PhysicalPropertyDataSet
            The secondary data set to merge into this one.
        """
        if data_set is None:
            return

        # TODO: Do we need to check whether merging the same data set here?
        for substance_hash in data_set.properties:

            if substance_hash not in self._properties:
                self._properties[substance_hash] = []

            self._properties[substance_hash].extend(data_set.properties[substance_hash])

        self._sources.extend(data_set.sources)

    def add_properties(self, *physical_properties):
        """Adds a physical property to the data set.

        Parameters
        ----------
        physical_properties: tuple of PhysicalProperty
            The physical property to add.
        """

        for physical_property in physical_properties:

            if physical_property.substance.identifier not in self._properties:
                self._properties[physical_property.substance.identifier] = []

            self._properties[physical_property.substance.identifier].append(
                physical_property
            )

    def filter_by_function(self, filter_function):
        """Filter the data set using a given filter function.

        Parameters
        ----------
        filter_function : lambda
            The filter function.
        """

        filtered_properties = {}

        # This works for now - if we wish to be able to undo a filter then
        # a 'filtered' list needs to be maintained separately to the main list.
        for substance_id in self._properties:

            substance_properties = list(
                filter(filter_function, self._properties[substance_id])
            )

            if len(substance_properties) <= 0:
                continue

            filtered_properties[substance_id] = substance_properties

        self._properties = {}

        for substance_id in filtered_properties:
            self._properties[substance_id] = filtered_properties[substance_id]

    def filter_by_property_types(self, *property_type):
        """Filter the data set based on the type of property (e.g Density).

        Parameters
        ----------
        property_type : PropertyType or str
            The type of property which should be retained.

        Examples
        --------
        Filter the dataset to only contain densities and static dielectric constants

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> # Filter the dataset to only include densities and dielectric constants.
        >>> from propertyestimator.properties import Density, DielectricConstant
        >>> data_set.filter_by_property_types(Density, DielectricConstant)

        or

        >>> data_set.filter_by_property_types('Density', 'DielectricConstant')
        """
        property_types = []

        for type_to_retain in property_type:

            if isinstance(type_to_retain, str):
                property_types.append(type_to_retain)
            else:
                property_types.append(type_to_retain.__name__)

        def filter_function(x):
            return type(x).__name__ in property_types

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
        >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from propertyestimator.datasets import PropertyPhase
        >>> data_set.filter_by_temperature(PropertyPhase.Liquid)
        """

        def filter_function(x):
            return x.phase & phases

        self.filter_by_function(filter_function)

    def filter_by_temperature(self, min_temperature, max_temperature):
        """Filter the data set based on a minimum and maximum temperature.

        Parameters
        ----------
        min_temperature : unit.Quantity
            The minimum temperature.
        max_temperature : unit.Quantity
            The maximum temperature.

        Examples
        --------
        Filter the dataset to only include properties measured between 130-260 K.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from propertyestimator import unit
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
        min_pressure : unit.Quantity
            The minimum pressure.
        max_pressure : unit.Quantity
            The maximum pressure.

        Examples
        --------
        Filter the dataset to only include properties measured between 70-150 kPa.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
        >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
        >>>
        >>> from propertyestimator import unit
        >>> data_set.filter_by_temperature(min_pressure=70*unit.kilopascal, max_temperature=150*unit.kilopascal)
        """

        def filter_function(x):

            if x.thermodynamic_state.pressure == UNDEFINED:
                return True

            return min_pressure <= x.thermodynamic_state.pressure <= max_pressure

        self.filter_by_function(filter_function)

    def filter_by_components(self, number_of_components):
        """Filter the data set based on a minimum and maximum temperature.

        Parameters
        ----------
        number_of_components : int
            The allowed number of components in the mixture.

        Examples
        --------
        Filter the dataset to only include pure substance properties.

        >>> # Load in the data set of properties which will be used for comparisons
        >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
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

        def filter_function(physical_property):

            substance = physical_property.substance

            for component in substance.components:

                oe_molecule = create_molecule_from_smiles(component.smiles, 0)

                for atom in oe_molecule.GetAtoms():

                    atomic_number = atom.GetAtomicNum()
                    atomic_element = element.Element.getByAtomicNumber(
                        atomic_number
                    ).symbol

                    if atomic_element in allowed_elements:
                        continue

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
            - 'Number Of Components'
            - 'Component 1'
            - 'Mole Fraction 1'
            - ...
            - 'Component N'
            - 'Mole Fraction N'
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
        # Determine the maximum number of components for any
        # given measurements.
        maximum_number_of_components = 0
        all_property_types = set()

        for substance_id in self._properties:

            if len(self._properties[substance_id]) == 0:
                continue

            substance = self._properties[substance_id][0].substance
            maximum_number_of_components = max(
                maximum_number_of_components, substance.number_of_components
            )

            for physical_property in self._properties[substance_id]:
                all_property_types.add(type(physical_property))

        # Make sure the maximum number of components is not zero.
        if maximum_number_of_components <= 0 < len(self._properties):

            raise ValueError(
                "The data set did not contain any substances with "
                "one or more components."
            )

        data_rows = []

        # Extract the data from the data set.
        for substance_id in self._properties:

            data_points_by_state = defaultdict(dict)

            for physical_property in self._properties[substance_id]:

                all_property_types.add(type(physical_property))

                # Extract the measured state.
                temperature = physical_property.thermodynamic_state.temperature.to(
                    unit.kelvin
                )
                pressure = None

                if physical_property.thermodynamic_state.pressure != UNDEFINED:
                    pressure = physical_property.thermodynamic_state.pressure.to(
                        unit.kilopascal
                    )

                phase = physical_property.phase

                # Extract the component data.
                number_of_components = physical_property.substance.number_of_components

                components = [] * maximum_number_of_components

                for index, component in enumerate(
                    physical_property.substance.components
                ):

                    amount = next(
                        iter(physical_property.substance.get_amounts(component))
                    )
                    assert isinstance(amount, MoleFraction)

                    components.append((component.smiles, amount.value))

                # Extract the value data as a string.
                value = (
                    None
                    if physical_property.value is None
                    else str(physical_property.value)
                )
                uncertainty = (
                    None
                    if physical_property.uncertainty is None
                    else str(physical_property.uncertainty)
                )

                # Extract the data source.
                source = None

                if isinstance(physical_property.source, MeasurementSource):

                    source = physical_property.source.reference

                    if source is None:
                        source = physical_property.source.doi

                elif isinstance(physical_property.source, CalculationSource):
                    source = physical_property.source.fidelity

                # Create the data row.
                data_row = {
                    "Temperature": str(temperature),
                    "Pressure": str(pressure),
                    "Phase": phase,
                    "Number Of Components": number_of_components,
                }

                for index in range(len(components)):

                    data_row[f"Component {index + 1}"] = components[index][0]
                    data_row[f"Mole Fraction {index + 1}"] = components[index][1]

                data_row[f"{type(physical_property).__name__} Value"] = value
                data_row[
                    f"{type(physical_property).__name__} Uncertainty"
                ] = uncertainty

                data_row["Source"] = source

                data_points_by_state[physical_property.thermodynamic_state].update(
                    data_row
                )

            for state in data_points_by_state:
                data_rows.append(data_points_by_state[state])

        # Set up the column headers.
        if len(data_rows) == 0:
            return None

        data_columns = [
            "Temperature",
            "Pressure",
            "Phase",
            "Number Of Components",
        ]

        for index in range(maximum_number_of_components):
            data_columns.append(f"Component {index + 1}")
            data_columns.append(f"Mole Fraction {index + 1}")

        for property_type in all_property_types:
            data_columns.append(f"{property_type.__name__} Value")
            data_columns.append(f"{property_type.__name__} Uncertainty")

        data_frame = pandas.DataFrame(data_rows, columns=data_columns)
        return data_frame

    def __len__(self):
        return self.number_of_properties

    def __getstate__(self):

        return {"properties": self._properties, "sources": self._sources}

    def __setstate__(self, state):

        self._properties = state["properties"]
        self._sources = state["sources"]

        for key in self._properties:
            assert all(isinstance(x, PhysicalProperty) for x in self._properties[key])

        assert all(isinstance(x, Source) for x in self._sources)
