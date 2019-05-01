"""
An API for defining and creating substances.
"""

from enum import Enum

from propertyestimator.utils.serialization import TypedBaseModel


class Substance(TypedBaseModel):
    """Defines the components, their amounts, and their roles in a system.

    Examples
    --------
    A neat liquid has only one component:

    >>> liquid = Substance()
    >>> liquid.add_component(Substance.Component(smiles='O'))

    A binary mixture has two components, where the mole fractions must be
    explicitly stated:

    >>> binary_mixture = Substance()
    >>> binary_mixture.add_component(Substance.Component(smiles='O'), mole_fraction=0.2)
    >>> binary_mixture.add_component(Substance.Component(smiles='CO'), mole_fraction=0.8)

    The infinite dilution of one solute within a solvent or mixture may also specified
    as a `Substance` by setting the mole fraction of the solute equal to 0.0.

    In this example we explicitly flag the benzene component as being the solute, and the
    water component the solvent, to aid in setting up and performing solvation free energy
    calculations:

    >>> benzene = Substance.Component(smiles='C1=CC=CC=C1', role=Substance.ComponentRole.Solute)
    >>> water = Substance.Component(smiles='O', role=Substance.ComponentRole.Solvent)

    >>> infinite_dilution = Substance()
    >>> infinite_dilution.add_component(component=benzene, mole_fraction=0.0) # Infinite dilution.
    >>> infinite_dilution.add_component(component=water, mole_fraction=1.0)
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

    @property
    def identifier(self):

        component_identifiers = [component.identifier for component in self._components]
        component_identifiers.sort()

        sorted_component_identifiers = [component.identifier for component in self._components]
        sorted_component_identifiers.sort()

        identifier_split = []

        for component_identifier in sorted_component_identifiers:

            component_fraction = self._mole_fractions[component_identifier]

            identifier = f'{component_identifier}'

            if component_fraction > 0.0:
                identifier += f'{{{component_fraction:.6f}}}'

            identifier_split.append(identifier)

        return '|'.join(identifier_split)

    @property
    def components(self):
        return self._components

    @property
    def number_of_components(self):
        return len(self._components)

    def __init__(self):
        """Constructs a new Substance object."""

        self._mole_fractions = {}
        self._components = []

    def add_component(self, component, mole_fraction=1.0):
        """Add a component to the Substance. If the component is already present in
        the substance, then the mole fraction will be added to the current mole
        fraction of that component.

        Parameters
        ----------
        component : Substance.Component
            The component to add to the system.
        mole_fraction : float
            The mole fraction of this component in the range of [0,1]. If a value of
            0.0 is provided, then the component will be treated as being infinitely
            dilute (i.e only present as a single molecule).
        """

        assert isinstance(component, Substance.Component)

        if mole_fraction < 0.0 or mole_fraction > 1.0:
            raise ValueError(f'The mole fraction ({mole_fraction} must be in the range [0.0, 1.0]')

        total_mole_fraction = mole_fraction + sum([mole_fraction for mole_fraction in
                                                   self._mole_fractions.values()])

        if total_mole_fraction > 1.0:
            raise ValueError(f'The total mole fraction of this substance {total_mole_fraction} exceeds 1.0')

        if component.identifier not in self._mole_fractions:
            self._mole_fractions[component.identifier] = 0.0

        self._mole_fractions[component.identifier] += mole_fraction
        self._components.append(component)

    def get_mole_fraction(self, component):
        """Returns the mole fraction of the component in this substance.

        Parameters
        ----------
        component: str or Substance.Component
            The component (or it's identifier) to retrieve the mole fraction of.

        Returns
        -------
        float
            The mole fraction of the component in this substance.
        """
        assert isinstance(component, str) or isinstance(component, Substance.Component)
        identifier = component if isinstance(component, str) else component.identifier

        return self._mole_fractions[identifier]

    def __getstate__(self):
        return {
            'components': self._components,
            'mole_fractions': self._mole_fractions
        }

    def __setstate__(self, state):
        self._components = state['components']
        self._mole_fractions = state['mole_fractions']

    def __str__(self):
        return self.identifier

    def __hash__(self):

        sorted_component_identifiers = [component.identifier for component in self._components]
        sorted_component_identifiers.sort()

        component_by_id = {component.identifier: component for component in self._components}

        string_hash_split = []

        for identifier in sorted_component_identifiers:

            component_role = component_by_id[identifier].role
            component_fraction = self._mole_fractions[identifier]

            string_hash_split.append(f'{identifier}_{component_role}_{component_fraction:.6f}')

        string_hash = '|'.join(string_hash_split)

        return hash(string_hash)

    def __eq__(self, other):

        return hash(self) == hash(other)

    def __ne__(self, other):
        return not (self == other)
