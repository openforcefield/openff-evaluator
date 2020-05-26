"""
An API for defining and creating substances.
"""

from enum import Enum

from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass


class Component(AttributeClass):
    """Defines a single component in a chemical system, as well
    as it's role within the system (if any).
    """

    class Role(Enum):
        """An enum which describes the role of a component in the system,
        such as whether the component is a solvent, a solute, a receptor etc.

        These roles are mainly used by workflow to identify the correct
        species in a system, such as when doing docking or performing
        solvation free energy calculations.
        """

        Solvent = "solv"
        Solute = "sol"

        Ligand = "lig"
        Receptor = "rec"

    smiles = Attribute(
        docstring="The SMILES pattern which describes this component.",
        type_hint=str,
        read_only=True,
    )
    role = Attribute(
        docstring="The role of this component in the system.",
        type_hint=Role,
        default_value=Role.Solvent,
        read_only=True,
    )

    @property
    def identifier(self):
        """str: A unique identifier for this component."""
        return f"{self.smiles}{{{self.role.value}}}"

    def __init__(self, smiles=UNDEFINED, role=Role.Solvent):
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
        role: Component.Role
            The role of this component in the system.
        """
        if smiles != UNDEFINED:
            smiles = self._standardize_smiles(smiles)

        self._set_value("smiles", smiles)
        self._set_value("role", role)

    @staticmethod
    def _standardize_smiles(smiles):
        """Standardizes a SMILES pattern to be canonical (but not necessarily isomeric)
        using the `cmiles` library.

        Parameters
        ----------
        smiles: str
            The SMILES pattern to standardize.

        Returns
        -------
        The standardized SMILES pattern.
        """
        from cmiles.utils import load_molecule, mol_to_smiles

        molecule = load_molecule(smiles, toolkit="rdkit")

        try:
            # Try to make the smiles isomeric.
            smiles = mol_to_smiles(
                molecule, isomeric=True, explicit_hydrogen=False, mapped=False
            )
        except ValueError:
            # Fall-back to non-isomeric.
            smiles = mol_to_smiles(
                molecule, isomeric=False, explicit_hydrogen=False, mapped=False
            )

        return smiles

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return f"<{self.__class__.__name__} {str(self)}>"

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return type(self) == type(other) and self.identifier == other.identifier

    def __ne__(self, other):
        return not (self == other)

    def __setstate__(self, state):
        # Make sure the smiles pattern is standardized.
        state["smiles"] = Component._standardize_smiles(state["smiles"])
        super(Component, self).__setstate__(state)
