"""
An API for defining and creating substances.
"""
from collections import abc

import numpy as np

from propertyestimator import unit
from propertyestimator.attributes import Attribute, AttributeClass
from propertyestimator.substances import Amount, Component, ExactAmount, MoleFraction


class Substance(AttributeClass):
    """Defines the components, their amounts, and their roles in a system.

    See Also
    --------
    physicalproperties
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
    def from_smiles(cls, *smiles, amounts=None):
        """Creates a new `Substance` object from a list of SMILES strings
        and optionally their amounts.

        Parameters
        ----------
        smiles: str
            The SMILES representation of the components to add to the substance.
        amounts: dict of str and Amount or iterable of Amount, optional
            The amount of each component being added. Each key should correspond to one
            of the passed `smiles`, and the value should be the amount(s) of that
            component. If `None`, it will be assumed each component is present with
            an equal mole fraction.

        Returns
        -------
        Substance
            The substance containing the requested components in equal amounts.

        See Also
        --------
        physicalproperties
        """

        if len(smiles) == 0:
            raise ValueError("At least one component must be specified")

        if not all(isinstance(x, str) for x in smiles):
            raise ValueError("The SMILES patterns must all be strings.")

        # Convert the smiles to components
        components = [Component(smiles=x) for x in smiles]

        if amounts is not None:

            if not all(x in amounts for x in smiles):
                raise ValueError("Each component must have a corresponding amount defined.")

            # Update the amounts dictionary to use the new components as keys.
            component_map = {x: y for x, y in zip(smiles, components)}
            amounts = {component_map[x]: amounts[x] for x in amounts}

        return cls.from_components(*components, amounts=amounts)

    @classmethod
    def from_components(cls, *components, amounts=None):
        """Creates a new `Substance` object from a list of components.

        Parameters
        ----------
        components: Component
            The components to add to the substance.
        amounts: dict of Component and Amount or iterable of Amount, optional
            The amount of each component being added. Each key should correspond to one
            of the passed `components`, and the value should be the amount(s) of that
            component. If `None`, it will be assumed each component is present with
            an equal mole fraction.

        Returns
        -------
        Substance
            The substance containing the requested components in equal amounts.

        See Also
        --------
        physicalproperties
        """

        if len(components) == 0:
            raise ValueError("At least one component must be specified")

        if not all(isinstance(x, Component) for x in components):
            raise ValueError("The components must be `Component` objects")

        # Make sure all of the components have at least one amount defined
        if amounts is None:

            mole_fraction = 1.0 / len(components)
            amounts = {x: MoleFraction(mole_fraction) for x in components}

        if not all(x in amounts for x in components):
            raise ValueError("Each component must have a corresponding amount defined.")

        # Validate the amounts.
        for amount_key, amount_value in amounts.items():

            if amount_key not in components:

                raise ValueError(
                    f"The amounts dictionary contained amounts for an undefined "
                    f"component ({amount_key})."
                )

            assert isinstance(amount_value, (Amount, abc.Iterable))

            if isinstance(amount_value, Amount):
                amount_value = [amount_value]

            assert all(isinstance(x, Amount) for x in amount_value)
            amounts[amount_key] = amount_value

        return_substance = cls()

        for component in components:
            for amount in amounts[component]:
                return_substance._add_component(component, amount)

        return return_substance

    def _add_component(self, component, amount):
        """Add a component to the Substance. If the component is already present in
        the substance, then the amount will be added to the current amount of that component.

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
        return ionic_mole_fraction

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
