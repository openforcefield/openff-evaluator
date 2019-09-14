"""
An API for defining and creating substances.
"""

import abc
import math
from enum import Enum

import numpy as np

from propertyestimator import unit
from propertyestimator.utils.serialization import TypedBaseModel


class Substance(TypedBaseModel):
    """Defines the components, their amounts, and their roles in a system.

    Examples
    --------
    A neat liquid containing only a single component:

    >>> liquid = Substance()
    >>> liquid.add_component(Substance.Component(smiles='O'), Substance.MoleFraction(1.0))

    A binary mixture containing two components, where the mole fractions are explicitly stated:

    >>> binary_mixture = Substance()
    >>> binary_mixture.add_component(Substance.Component(smiles='O'), Substance.MoleFraction(0.2))
    >>> binary_mixture.add_component(Substance.Component(smiles='CO'), Substance.MoleFraction(0.8))

    The infinite dilution of one molecule within a bulk solvent or mixture may also be specified
    by defining the exact number of copies of that molecule, rather than a mole fraction:

    >>> benzene = Substance.Component(smiles='C1=CC=CC=C1', role=Substance.ComponentRole.Solute)
    >>> water = Substance.Component(smiles='O', role=Substance.ComponentRole.Solvent)
    >>>
    >>> infinite_dilution = Substance()
    >>> infinite_dilution.add_component(component=benzene, amount=Substance.ExactAmount(1)) # Infinite dilution.
    >>> infinite_dilution.add_component(component=water, amount=Substance.MoleFraction(1.0))

    In this example we explicitly flag benzene as being the solute and the water component the solvent.
    This enables workflow's to easily identify key molecules of interest, such as the molecule which should
    be 'grown' into solution during solvation free energy calculations.
    """

    class ComponentRole(Enum):
        """An enum which describes the role of a component in the system,
        such as whether the component is a solvent, a solute, a receptor etc.

        These roles are mainly only used by specific protocols to identify
        the correct species in a system, such as when doing docking or performing
        solvation free energy calculations.
        """
        Solvent = 'Solvent'
        Solute = 'Solute'

        Ligand = 'Ligand'
        Receptor = 'Receptor'

        Undefined = 'Undefined'

    class Component(TypedBaseModel):
        """Defines a single component in a system, as well as properties
        such as it's relative proportion in the system.
        """

        @property
        def identifier(self):
            """str: A unique identifier for this component, which is either a
            smiles descriptor or the supplied label."""
            return self._smiles or self._label

        @property
        def label(self):
            """str: A string label which describes this compound, for example, CB8."""
            return self._label

        @property
        def smiles(self):
            """str: The smiles pattern which describes this component, which may be None
            for complex (e.g protein) molecules."""
            return self._smiles

        @property
        def role(self):
            """ComponentRole: The role of this component in the system, such as a
            ligand or a receptor."""
            return self._role

        def __init__(self, smiles=None, label=None, role=None):
            """Constructs a new Component object with either a label or
            a smiles string, but not both.

            Notes
            -----
            The `label` and `smiles` arguments are mutually exclusive, and only
            one can be passed while the other should be `None`.

            Parameters
            ----------
            smiles: str
                A SMILES descriptor of the component
            label: str
                A string label which describes this compound, for example, CB8.
            role: ComponentRole, optional
                The role of this component in the system. If no role is specified,
                a default role of solvent is applied.
            """

            if label == smiles:
                label = None

            assert ((label is None and smiles is not None) or
                    (label is not None and smiles is None) or
                    (label is None and smiles is None))

            label = label if label is not None else smiles

            self._label = label
            self._smiles = smiles

            self._role = role or Substance.ComponentRole.Solvent

        def __getstate__(self):
            return {
                'label': self.label,
                'smiles': self.smiles,

                'role': self.role
            }

        def __setstate__(self, state):
            self._label = state['label']
            self._smiles = state['smiles']

            self._role = state['role']

        def __str__(self):
            return self.identifier

        def __hash__(self):
            return hash((self.identifier, self._role))

        def __eq__(self, other):
            return hash(self) == hash(other)

        def __ne__(self, other):
            return not (self == other)

    class Amount(abc.ABC):
        """An abstract representation of the amount of a given component
        in a substance.
        """

        @property
        def value(self):
            """The value of this amount."""
            return self._value

        @property
        def identifier(self):
            """A string identifier for this amount."""
            raise NotImplementedError()

        def __init__(self, value=None):
            """Constructs a new Amount object."""
            self._value = value

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

        def __getstate__(self):
            return {'value': self._value}

        def __setstate__(self, state):
            self._value = state['value']

        def __str__(self):
            return self.identifier

        def __eq__(self, other):
            return np.isclose(self._value, other.value)

        def __ne__(self, other):
            return not (self == other)

        def __hash__(self):
            return hash(self.identifier)

    class MoleFraction(Amount):
        """Represents the amount of a component in a substance as a
        mole fraction."""

        @property
        def value(self):
            """float: The value of this amount."""
            return super(Substance.MoleFraction, self).value

        @property
        def identifier(self):
            return f'{{{self._value:.6f}}}'

        def __init__(self, value=1.0):
            """Constructs a new MoleFraction object.

            Parameters
            ----------
            value: float
                A mole fraction in the range (0.0, 1.0]
            """

            if value <= 0.0 or value > 1.0:

                raise ValueError('A mole fraction must be greater than zero, and less than or '
                                 'equal to one.')

            if math.floor(value * 1e6) < 1:

                raise ValueError('Mole fractions are only precise to the sixth '
                                 'decimal place within this class representation.')

            super().__init__(value)

        def to_number_of_molecules(self, total_substance_molecules, tolerance=None):

            # Determine how many molecules of each type will be present in the system.
            number_of_molecules = self._value * total_substance_molecules
            fractional_number_of_molecules = number_of_molecules % 1

            if np.isclose(fractional_number_of_molecules, 0.5):
                number_of_molecules = int(number_of_molecules)
            else:
                number_of_molecules = int(round(number_of_molecules))

            if number_of_molecules == 0:
                raise ValueError('The total number of substance molecules was not large enough, '
                                 'such that this non-zero amount translates into zero molecules '
                                 'of this component in the substance.')

            if tolerance is not None:

                mole_fraction = number_of_molecules / total_substance_molecules

                if abs(mole_fraction - self._value) > tolerance:
                    raise ValueError(f'The mole fraction ({mole_fraction}) given a total number of molecules '
                                     f'({total_substance_molecules}) is outside of the tolerance {tolerance} '
                                     f'of the target mole fraction {self._value}')

            return number_of_molecules

    class ExactAmount(Amount):
        """Represents the amount of a component in a substance as an
        exact number of molecules.

        The expectation is that this amount should be used for components which
        are infinitely dilute (such as ligands in binding calculations), and hence
        do not contribute to the total mole fraction of a substance"""

        @property
        def value(self):
            """int: The value of this amount."""
            return super(Substance.ExactAmount, self).value

        @property
        def identifier(self):
            return f'({int(round(self._value)):d})'

        def __init__(self, value=1):
            """Constructs a new ExactAmount object.

            Parameters
            ----------
            value: int
                An exact number of molecules.
            """

            if not np.isclose(int(round(value)), value):
                raise ValueError('The value must be an integer.')

            super().__init__(value)

        def to_number_of_molecules(self, total_substance_molecules, tolerance=None):
            return self._value

    @property
    def identifier(self):
        """str: A unique str representation of this substance, which encodes all components
        and their amounts in the substance."""

        component_identifiers = [component.identifier for component in self._components]
        component_identifiers.sort()

        sorted_component_identifiers = [component.identifier for component in self._components]
        sorted_component_identifiers.sort()

        identifier_split = []

        for component_identifier in sorted_component_identifiers:

            component_amounts = sorted(self._amounts[component_identifier], key=lambda x: type(x).__name__)
            amount_identifier = ''.join([component_amount.identifier for component_amount in component_amounts])

            identifier = f'{component_identifier}{amount_identifier}'
            identifier_split.append(identifier)

        return '|'.join(identifier_split)

    @property
    def components(self):
        """list of Substance.Component: A list of all of the components in this substance."""
        return self._components

    @property
    def number_of_components(self):
        """int: The number of different components in this substance."""
        return len(self._components)

    def __init__(self):
        """Constructs a new Substance object."""

        self._amounts = {}
        self._components = []

    @classmethod
    def from_components(cls, *components):
        """Creates a new `Substance` object from a list of components.
        This method assumes that all components should be present with
        equal mole fractions.

        Parameters
        ----------
        components: Substance.Component or str
            The components to add to the substance. These may either be full
            `Substance.Component` objects or just the smiles representation
            of the component.

        Returns
        -------
        Substance
            The substance containing the requested components in equal amounts.
        """

        if len(components) == 0:
            raise ValueError('At least one component must be specified')

        mole_fraction = 1.0 / len(components)

        return_substance = cls()

        for component in components:

            if isinstance(component, str):
                component = Substance.Component(smiles=component)

            return_substance.add_component(component, Substance.MoleFraction(mole_fraction))

        return return_substance

    def add_component(self, component, amount):
        """Add a component to the Substance. If the component is already present in
        the substance, then the mole fraction will be added to the current mole
        fraction of that component.

        Parameters
        ----------
        component : Substance.Component
            The component to add to the system.
        amount : Substance.Amount
            The amount of this component in the substance.
        """

        assert isinstance(component, Substance.Component)
        assert isinstance(amount, Substance.Amount)

        if isinstance(amount, Substance.MoleFraction):

            total_mole_fraction = amount.value

            for component_identifier in self._amounts:

                total_mole_fraction += sum([amount.value for amount in self._amounts[component_identifier] if
                                            isinstance(amount, Substance.MoleFraction)])

            if np.isclose(total_mole_fraction, 1.0):
                total_mole_fraction = 1.0

            if total_mole_fraction > 1.0:
                raise ValueError(f'The total mole fraction of this substance {total_mole_fraction} exceeds 1.0')

        if component.identifier not in self._amounts:
            self._components.append(component)

        existing_amount_of_type = None

        all_amounts = [] if component.identifier not in self._amounts else self._amounts[component.identifier]
        remaining_amounts = []

        # Check to see if an amount of the same type already exists in
        # the substance, such that this amount should be appended to it.
        for existing_amount in all_amounts:

            if not type(existing_amount) is type(amount):

                remaining_amounts.append(existing_amount)
                continue

            existing_amount_of_type = existing_amount
            break

        if existing_amount_of_type is not None:

            # Append any existing amounts to the new amount.
            amount = type(amount)(existing_amount_of_type.value + amount.value)

        remaining_amounts.append(amount)
        self._amounts[component.identifier] = frozenset(remaining_amounts)

    def get_amounts(self, component):
        """Returns the amounts of the component in this substance.

        Parameters
        ----------
        component: str or Substance.Component
            The component (or it's identifier) to retrieve the amount of.

        Returns
        -------
        list of Substance.Amount
            The amounts of the component in this substance.
        """
        assert isinstance(component, str) or isinstance(component, Substance.Component)
        identifier = component if isinstance(component, str) else component.identifier

        return self._amounts[identifier]

    def get_molecules_per_component(self, maximum_molecules, tolerance=None):
        """Returns the number of molecules for each component in this substance,
        given a maximum total number of molecules.

        Parameters
        ----------
        maximum_molecules: int
            The maximum number of molecules.
        tolerance: float, optional
            The tolerance within which this amount should be represented. As
            an example, when converting a mole fraction into a number of molecules,
            the total number of molecules may not be sufficiently large enough to
            reproduce this amount.

        Returns
        -------
        dict of str and int
            A dictionary of molecule counts per component, where each key is
            a component identifier.
        """

        number_of_molecules = {}
        remaining_molecule_slots = maximum_molecules

        for index, component in enumerate(self._components):

            amounts = self._amounts[component.identifier]

            for amount in amounts:

                if not isinstance(amount, Substance.ExactAmount):
                    continue

                remaining_molecule_slots -= amount.value

        if remaining_molecule_slots < 0:

            raise ValueError(f'The required number of molecules {maximum_molecules - remaining_molecule_slots} '
                             f'exceeds the provided maximum number ({maximum_molecules}).')

        for component in self._components:

            number_of_molecules[component.identifier] = 0

            for amount in self._amounts[component.identifier]:

                number_of_molecules[component.identifier] += amount.to_number_of_molecules(remaining_molecule_slots,
                                                                                           tolerance)

        return number_of_molecules

    @staticmethod
    def calculate_aqueous_ionic_mole_fraction(ionic_strength):
        """Determines what mole fraction of ions is needed to yield
         an aqueous system of a given ionic strength.

        Parameters
        ----------
        ionic_strength: unit.Quantity
            The ionic string in units of molar.

        Returns
        -------
        float
            The mole fraction of ions.
        """

        # Taken from YANK:
        # https://github.com/choderalab/yank/blob/4dfcc8e127c51c20180fe6caeb49fcb1f21730c6/Yank/pipeline.py#L1869
        water_molarity = (998.23 * unit.gram / unit.litre) / (18.01528 * unit.gram / unit.mole)

        ionic_mole_fraction = ionic_strength / (ionic_strength + water_molarity)
        return ionic_mole_fraction

    def __getstate__(self):
        return {
            'components': self._components,
            'amounts': self._amounts
        }

    def __setstate__(self, state):
        self._components = state['components']
        self._amounts = state['amounts']

    def __str__(self):
        return self.identifier

    def __hash__(self):

        sorted_component_identifiers = [component.identifier for component in self._components]
        sorted_component_identifiers.sort()

        component_by_id = {component.identifier: component for component in self._components}

        string_hash_split = []

        for identifier in sorted_component_identifiers:

            component_role = component_by_id[identifier].role

            component_amounts = sorted(self._amounts[identifier], key=lambda x: type(x).__name__)
            amount_identifier = ''.join([component_amount.identifier for component_amount in component_amounts])

            string_hash_split.append(f'{identifier}_{component_role}_{amount_identifier}')

        string_hash = '|'.join(string_hash_split)

        return hash(string_hash)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not (self == other)
