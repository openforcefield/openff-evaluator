"""
An API for defining and creating substances.
"""
import numpy as np

from openff.evaluator import unit
from openff.evaluator.attributes import Attribute, AttributeClass
from openff.evaluator.substances import Amount, Component, ExactAmount, MoleFraction


class Substance(AttributeClass):
    """Defines the components, their amounts, and their roles in a system.

    Examples
    --------
    A neat liquid containing only a single component:

    >>> from openff.evaluator.substances import Component, ExactAmount, MoleFraction
    >>> liquid = Substance()
    >>> liquid.add_component(Component(smiles='O'), MoleFraction(1.0))

    A binary mixture containing two components, where the mole fractions are explicitly stated:

    >>> binary_mixture = Substance()
    >>> binary_mixture.add_component(Component(smiles='O'), MoleFraction(0.2))
    >>> binary_mixture.add_component(Component(smiles='CO'), MoleFraction(0.8))

    The infinite dilution of one molecule within a bulk solvent or mixture may also be specified
    by defining the exact number of copies of that molecule, rather than a mole fraction:

    >>> benzene = Component(smiles='C1=CC=CC=C1', role=Component.Role.Solute)
    >>> water = Component(smiles='O', role=Component.Role.Solvent)
    >>>
    >>> infinite_dilution = Substance()
    >>> infinite_dilution.add_component(component=benzene, amount=ExactAmount(1)) # Infinite dilution.
    >>> infinite_dilution.add_component(component=water, amount=MoleFraction(1.0))

    In this example we explicitly flag benzene as being the solute and the water component the solvent.
    This enables workflow's to easily identify key molecules of interest, such as the molecule which should
    be 'grown' into solution during solvation free energy calculations.
    """

    components = Attribute(
        docstring="A list of all of the components in this substance.",
        type_hint=tuple,
        default_value=tuple(),
        read_only=True,
    )
    amounts = Attribute(
        docstring="the amounts of the component in this substance",
        type_hint=dict,
        default_value=dict(),
        read_only=True,
    )

    @property
    def identifier(self):
        """str: A unique str representation of this substance, which encodes all
        components and their amounts in the substance."""
        return self._get_identifier()

    @property
    def number_of_components(self):
        """int: The number of different components in this substance."""
        return len(self.components)

    def _get_identifier(self):
        """Generates a unique string identifier for this substance, which
        encodes all components and their amounts in the substance

        Returns
        -------
        str
            The string identifier.
        """
        component_identifiers = [component.identifier for component in self.components]
        component_identifiers.sort()

        identifier_split = []

        for component_identifier in component_identifiers:

            component_amounts = sorted(
                self.amounts[component_identifier], key=lambda x: type(x).__name__
            )
            amount_identifier = ",".join(
                [component_amount.identifier for component_amount in component_amounts]
            )

            identifier = f"{component_identifier}{{{amount_identifier}}}"
            identifier_split.append(identifier)

        return "|".join(identifier_split)

    @classmethod
    def from_components(cls, *components):
        """Creates a new `Substance` object from a list of components.
        This method assumes that all components should be present with
        equal mole fractions.

        Parameters
        ----------
        components: Component or str
            The components to add to the substance. These may either be full
            `Component` objects or just the smiles representation
            of the component.

        Returns
        -------
        Substance
            The substance containing the requested components in equal amounts.
        """

        if len(components) == 0:
            raise ValueError("At least one component must be specified")

        mole_fraction = 1.0 / len(components)

        return_substance = cls()

        for component in components:

            if isinstance(component, str):
                component = Component(smiles=component)

            return_substance.add_component(component, MoleFraction(mole_fraction))

        return return_substance

    def add_component(self, component, amount):
        """Add a component to the Substance. If the component is already present in
        the substance, then the mole fraction will be added to the current mole
        fraction of that component.

        Parameters
        ----------
        component : Component
            The component to add to the system.
        amount : Amount
            The amount of this component in the substance.
        """

        assert isinstance(component, Component)
        assert isinstance(amount, Amount)

        component.validate()
        amount.validate()

        if isinstance(amount, MoleFraction):

            total_mole_fraction = amount.value

            for component_identifier in self.amounts:

                total_mole_fraction += sum(
                    [
                        amount.value
                        for amount in self.amounts[component_identifier]
                        if isinstance(amount, MoleFraction)
                    ]
                )

            if np.isclose(total_mole_fraction, 1.0):
                total_mole_fraction = 1.0

            if total_mole_fraction > 1.0:

                raise ValueError(
                    f"The total mole fraction of this substance {total_mole_fraction} exceeds 1.0"
                )

        if component.identifier not in self.amounts:

            components = (*self.components, component)
            self._set_value("components", components)

        existing_amount_of_type = None

        all_amounts = (
            []
            if component.identifier not in self.amounts
            else self.amounts[component.identifier]
        )
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

        amounts = dict(self.amounts)
        amounts[component.identifier] = tuple(remaining_amounts)

        self._set_value("amounts", amounts)

    def get_amounts(self, component):
        """Returns the amounts of the component in this substance.

        Parameters
        ----------
        component: str or Component
            The component (or it's identifier) to retrieve the amount of.

        Returns
        -------
        tuple of Amount
            The amounts of the component in this substance.
        """
        assert isinstance(component, str) or isinstance(component, Component)
        identifier = component if isinstance(component, str) else component.identifier

        return self.amounts[identifier]

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

        for index, component in enumerate(self.components):

            amounts = self.amounts[component.identifier]

            for amount in amounts:

                if not isinstance(amount, ExactAmount):
                    continue

                remaining_molecule_slots -= amount.value

        if remaining_molecule_slots < 0:

            raise ValueError(
                f"The required number of molecules {maximum_molecules - remaining_molecule_slots} "
                f"exceeds the provided maximum number ({maximum_molecules})."
            )

        for component in self.components:

            number_of_molecules[component.identifier] = 0

            for amount in self.amounts[component.identifier]:

                number_of_molecules[
                    component.identifier
                ] += amount.to_number_of_molecules(remaining_molecule_slots, tolerance)

        return number_of_molecules

    @staticmethod
    def calculate_aqueous_ionic_mole_fraction(ionic_strength):
        """Determines what mole fraction of ions is needed to yield
         an aqueous system of a given ionic strength.

        Parameters
        ----------
        ionic_strength: pint.Quantity
            The ionic string in units of molar.

        Returns
        -------
        float
            The mole fraction of ions.
        """

        # Taken from YANK:
        # https://github.com/choderalab/yank/blob/4dfcc8e127c51c20180fe6caeb49fcb1f21730c6/Yank/pipeline.py#L1869
        water_molarity = (998.23 * unit.gram / unit.litre) / (
            18.01528 * unit.gram / unit.mole
        )

        ionic_mole_fraction = ionic_strength / (ionic_strength + water_molarity)
        return ionic_mole_fraction.magnitude

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return f"<Substance {str(self)}>"

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not (self == other)

    def __setstate__(self, state):
        # Handle the list -> tuple conversion manually.

        assert "amounts" in state

        for key in state["amounts"]:

            assert isinstance(state["amounts"][key], (list, tuple))
            state["amounts"][key] = tuple(state["amounts"][key])

        super(Substance, self).__setstate__(state)

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def validate(self, attribute_type=None):
        super(Substance, self).validate(attribute_type)

        # Validate all of the components.
        assert all(isinstance(x, Component) for x in self.components)
        assert all(x.identifier in self.amounts for x in self.components)

        # Validate the amounts
        assert all(x.identifier in self.amounts for x in self.components)
        assert all(isinstance(x, tuple) for x in self.amounts.values())
        assert all(len(x) > 0 for x in self.amounts.values())

        for component in self.components:

            component.validate(attribute_type)
            amounts = self.amounts[component.identifier]

            assert all(isinstance(x, Amount) for x in amounts)

            for amount in amounts:
                amount.validate(attribute_type)

        contains_mole_fraction = any(
            isinstance(x, MoleFraction) for y in self.amounts.values() for x in y
        )

        if contains_mole_fraction:

            total_mole_fraction = 0.0

            for component_identifier in self.amounts:

                total_mole_fraction += sum(
                    [
                        amount.value
                        for amount in self.amounts[component_identifier]
                        if isinstance(amount, MoleFraction)
                    ]
                )

            if not np.isclose(total_mole_fraction, 1.0):

                raise ValueError(
                    f"The total mole fraction of this substance "
                    f"({total_mole_fraction}) must equal 1.0"
                )
