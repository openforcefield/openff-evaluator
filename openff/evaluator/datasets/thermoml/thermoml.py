"""
An API for importing a ThermoML archive.
"""
import copy
import logging
import re
import traceback
from enum import Enum, unique
from urllib.error import HTTPError
from xml.etree import ElementTree

import numpy as np
import requests
from openff.units import unit

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.openmm import openmm_quantity_to_pint

logger = logging.getLogger(__name__)


def _unit_from_thermoml_string(full_string):
    """Extract the unit from a ThermoML property name.

    Parameters
    ----------
    full_string: str
        The string to convert to a Unit object.

    Returns
    ----------
    openff.evaluator.unit.Unit
        The parsed unit.
    """

    full_string_split = full_string.split(",")

    # Extract the unit portion of the string
    unit_string = full_string_split[1] if len(full_string_split) > 1 else ""
    # Convert symbols like dm3 to dm**3
    unit_string = re.sub(r"([a-z])([0-9]+)", r"\1**\2", unit_string.strip())

    return unit.Unit(unit_string)


def _phase_from_thermoml_string(string):
    """Converts a ThermoML string to a PropertyPhase

    Parameters
    ----------
    string: str
        The string to convert to a PropertyPhase

    Returns
    ----------
    PropertyPhase
        The converted PropertyPhase
    """
    phase_string = string.lower().strip()
    phase = PropertyPhase.Undefined

    if phase_string == "liquid" or phase_string.find("solution") >= 0:
        phase = PropertyPhase.Liquid
    elif phase_string.find("crystal") >= 0 and not phase_string.find("liquid") >= 0:
        phase = PropertyPhase.Solid
    elif phase_string.find("gas") >= 0:
        phase = PropertyPhase.Gas

    return phase


@unique
class _ConstraintType(Enum):
    """An enumeration of the supported types of ThermoML constraint
    types.
    """

    Undefined = "Undefined"
    Temperature = "Temperature, K"
    Pressure = "Pressure, kPa"
    ComponentMoleFraction = "Mole fraction"
    ComponentMassFraction = "Mass fraction"
    ComponentMolality = "Molality, mol/kg"
    SolventMoleFraction = "Solvent: Mole fraction"
    SolventMassFraction = "Solvent: Mass fraction"
    SolventMolality = "Solvent: Molality, mol/kg"

    @staticmethod
    def from_node(node):
        """Converts either a ConstraintType or VariableType xml node to a _ConstraintType.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to convert.

        Returns
        ----------
        _ConstraintType
            The converted constraint type.
        """

        try:
            constraint_type = _ConstraintType(node.text)
        except (KeyError, ValueError):
            constraint_type = _ConstraintType.Undefined

        if constraint_type == _ConstraintType.Undefined:
            logging.debug(f"{node.tag}->{node.text} is an unsupported constraint type.")

        return constraint_type

    def is_composition_constraint(self):
        """Checks whether the purpose of this constraint is
        to constrain the substance composition.

        Returns
        -------
        bool
            True if the constraint type is either a

            - `_ConstraintType.ComponentMoleFraction`
            - `_ConstraintType.ComponentMassFraction`
            - `_ConstraintType.ComponentMolality`
            - `_ConstraintType.SolventMoleFraction`
            - `_ConstraintType.SolventMassFraction`
            - `_ConstraintType.SolventMolality`
        """
        return (
            self == _ConstraintType.ComponentMoleFraction
            or self == _ConstraintType.ComponentMassFraction
            or self == _ConstraintType.ComponentMolality
            or self == _ConstraintType.SolventMoleFraction
            or self == _ConstraintType.SolventMassFraction
            or self == _ConstraintType.SolventMolality
        )


class _Constraint:
    """A wrapper around a ThermoML `Constraint` node. A constraint
    in ThermoML encompasses such constructs as temperature, pressure
    or composition at which a measurement was recorded.
    """

    def __init__(self):

        self.type = _ConstraintType.Undefined
        self.value = 0.0

        self.solvents = []

        # Describes which compound the variable acts upon.
        self.compound_index = None

    @classmethod
    def from_node(cls, constraint_node, namespace):
        """Creates a _Constraint from an xml node.

        Parameters
        ----------
        constraint_node: Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.

        Returns
        ----------
        _Constraint, optional
            The extracted constraint if the constraint type is supported,
            otherwise `None`.
        """
        # Extract the xml nodes.
        type_node = constraint_node.find(".//ThermoML:ConstraintType/*", namespace)
        value_node = constraint_node.find("./ThermoML:nConstraintValue", namespace)

        solvent_index_nodes = constraint_node.find(
            "./ThermoML:Solvent//ThermoML:nOrgNum", namespace
        )
        compound_index_node = constraint_node.find(
            "./ThermoML:ConstraintID/ThermoML:RegNum/*", namespace
        )

        value = float(value_node.text)
        # Determine what the default unit for this variable should be.
        unit_type = _unit_from_thermoml_string(type_node.text)

        return_value = cls()

        return_value.type = _ConstraintType.from_node(type_node)
        return_value.value = value * unit_type

        if compound_index_node is not None:
            return_value.compound_index = int(compound_index_node.text)

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node))

        return None if return_value.type is _ConstraintType.Undefined else return_value

    @classmethod
    def from_variable(cls, variable, value):
        """Creates a _Constraint from an existing
        `_VariableDefinition` variable definition.

        Parameters
        ----------
        variable: _VariableDefinition
            The variable to convert.
        value: openff.evaluator.unit.Quantity
            The value of the constant.

        Returns
        ----------
        _Constraint
            The created constraint.
        """
        return_value = cls()

        return_value.type = variable.type
        return_value.compound_index = variable.compound_index

        return_value.solvents.extend(variable.solvents)

        return_value.value = value

        return return_value


class _VariableDefinition:
    """A wrapper around a ThermoML Variable node. A variable in
    ThermoML is essentially just the definition of a `Constraint`
    (the constraint type, the expected units, etc.) whose value is
    defined inside of another ThermoML node.
    """

    def __init__(self):

        self.index = -1
        self.type = _ConstraintType.Undefined

        self.solvents = []
        self.default_unit = None

        # Describes which compound the variable acts upon.
        self.compound_index = None

    @classmethod
    def from_node(cls, variable_node, namespace):
        """Creates a `_VariableDefinition` from an xml node.

        Parameters
        ----------
        variable_node: xml.etree.Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.

        Returns
        ----------
        _VariableDefinition
            The created variable definition.
        """
        # Extract the xml nodes.
        type_node = variable_node.find(".//ThermoML:VariableType/*", namespace)
        index_node = variable_node.find("ThermoML:nVarNumber", namespace)

        solvent_index_nodes = variable_node.find(
            "./ThermoML:Solvent//ThermoML:nOrgNum", namespace
        )
        compound_index_node = variable_node.find(
            "./ThermoML:VariableID/ThermoML:RegNum/*", namespace
        )

        return_value = cls()

        return_value.default_unit = _unit_from_thermoml_string(type_node.text)

        return_value.index = int(index_node.text)
        return_value.type = _ConstraintType.from_node(type_node)

        if compound_index_node is not None:
            return_value.compound_index = int(compound_index_node.text)

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node))

        return None if return_value.type is _ConstraintType.Undefined else return_value


class _PropertyUncertainty:
    """A wrapper around a ThermoML PropUncertainty node."""

    # Reduce code redundancy by reusing this class for
    # both property and combined uncertainties.
    prefix = ""

    def __init__(self):

        self.index = -1
        self.coverage_factor = None

    @classmethod
    def from_xml(cls, node, namespace):
        """Creates a _PropertyUncertainty from an xml node.

        Parameters
        ----------
        node: Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.

        Returns
        ----------
        _Compound
            The created property uncertainty.
        """

        coverage_factor_node = node.find(
            f"ThermoML:n{cls.prefix}CoverageFactor", namespace
        )
        confidence_node = node.find(
            f"ThermoML:n{cls.prefix}UncertLevOfConfid", namespace
        )

        # As defined by https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-7-reporting-uncertainty
        if coverage_factor_node is not None:
            coverage_factor = float(coverage_factor_node.text)
        elif confidence_node is not None and confidence_node.text == "95":
            coverage_factor = 2
        else:
            return None

        index_node = node.find(f"ThermoML:n{cls.prefix}UncertAssessNum", namespace)
        index = int(index_node.text)

        return_value = cls()

        return_value.coverage_factor = coverage_factor
        return_value.index = index

        return return_value


class _CombinedUncertainty(_PropertyUncertainty):
    """A wrapper around a ThermoML CombPropUncertainty node."""

    prefix = "Comb"


class _Compound:
    """A wrapper around a ThermoML Compound node."""

    def __init__(self):

        self.smiles = None
        self.index = -1

    @staticmethod
    def smiles_from_inchi_string(inchi_string):
        """Attempts to create a SMILES pattern from an inchi string.

        Parameters
        ----------
        inchi_string: str
            The InChI string to convert.

        Returns
        ----------
        str, optional
            None if the identifier cannot be converted, otherwise the converted SMILES pattern.
        """
        from cmiles.utils import mol_to_smiles

        try:
            import rdkit.Chem
        except ImportError:
            return None

        if inchi_string is None:
            raise ValueError("The InChI string cannot be `None`.")

        molecule = rdkit.Chem.MolFromInchi(inchi_string, removeHs=False)

        if not molecule:
            raise ValueError(f"The InchI string ({inchi_string}) could not be parsed")

        try:
            return mol_to_smiles(molecule, explicit_hydrogen=False, mapped=False)
        except ValueError:
            return None

    @staticmethod
    def smiles_from_thermoml_smiles_string(thermoml_string):
        """Attempts to create a SMILES pattern from a thermoml smiles string.

        Parameters
        ----------
        thermoml_string: str
            The string to convert.

        Returns
        ----------
        str, optional
            None if the identifier cannot be converted, otherwise the converted SMILES pattern.
        """
        from cmiles.utils import load_molecule, mol_to_smiles

        if thermoml_string is None:
            raise ValueError("The string cannot be `None`.")

        molecule = load_molecule(thermoml_string, toolkit="rdkit")
        return mol_to_smiles(molecule, explicit_hydrogen=False, mapped=False)

    @staticmethod
    def smiles_from_common_name(common_name):
        """Attempts to create a SMILES pattern from an IUPAC name.

        Parameters
        ----------
        common_name: str
            The common name to convert.
        Returns
        ----------
        str, None
            None if the identifier cannot be converted, otherwise the converted SMILES pattern.
        """
        from cmiles.utils import load_molecule, mol_to_smiles
        from openff.toolkit.topology import Molecule
        from openff.toolkit.utils import InvalidIUPACNameError, LicenseError

        if common_name is None:
            return None

        try:

            molecule = Molecule.from_iupac(common_name, allow_undefined_stereo=True)
            cmiles_molecule = load_molecule(molecule.to_smiles(), toolkit="rdkit")
            smiles = mol_to_smiles(
                cmiles_molecule, explicit_hydrogen=False, mapped=False
            )

            if isinstance(smiles, str) and len(smiles) == 0:
                smiles = None

        except LicenseError:
            smiles = None
        except (ValueError, InvalidIUPACNameError):
            smiles = None

        return smiles

    @classmethod
    def from_xml_node(cls, node, namespace):
        """Creates a _Compound from an xml node.

        Parameters
        ----------
        node: Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.

        Returns
        ----------
        _Compound
            The created compound wrapper.
        """
        # Gather up all possible identifiers
        inchi_identifier_nodes = node.findall("ThermoML:sStandardInChI", namespace)
        smiles_identifier_nodes = node.findall("ThermoML:sSmiles", namespace)
        common_identifier_nodes = node.findall("ThermoML:sCommonName", namespace)

        smiles = None

        if (
            len(inchi_identifier_nodes) > 0
            and inchi_identifier_nodes[0].text is not None
        ):
            # Convert InChI key to a smiles pattern.
            smiles = cls.smiles_from_inchi_string(inchi_identifier_nodes[0].text)

        if (
            smiles is None
            and len(smiles_identifier_nodes) > 0
            and smiles_identifier_nodes[0].text is not None
        ):
            # Standardise the smiles pattern using OE.
            smiles = cls.smiles_from_thermoml_smiles_string(
                smiles_identifier_nodes[0].text
            )

        if (
            smiles is None
            and len(common_identifier_nodes) > 0
            and common_identifier_nodes[0].text is not None
        ):
            # Convert the common name to a smiles pattern.
            smiles = cls.smiles_from_common_name(common_identifier_nodes[0].text)

        if smiles is None:

            logging.debug(
                "A ThermoML:Compound node does not have a valid InChI identifier, "
                "a valid SMILES pattern, or an understandable common name."
            )

            return None

        index_node = node.find("./ThermoML:RegNum/*", namespace)

        if index_node is None:
            raise ValueError("A ThermoML:Compound does not have an index defined.")

        compound_index = int(index_node.text)

        return_value = cls()

        return_value.smiles = smiles
        return_value.index = compound_index

        return return_value


class _PureOrMixtureData:
    """A wrapper around a ThermoML PureOrMixtureData node."""

    @staticmethod
    def extract_compound_indices(node, namespace, compounds):
        """Extract a list of the compound indices which a given `PureOrMixtureData`
        node depends upon.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        compounds: dict of int and _Compound
            The compounds which were able to be parsed from the
            parent archive file, with keys of their assigned
            indices.
        """

        component_nodes = node.findall("ThermoML:Component", namespace)
        compound_indices = []

        # Figure out which compounds are going to be associated with
        # the property entries.
        for component_node in component_nodes:

            index_node = component_node.find("./ThermoML:RegNum/*", namespace)

            compound_index = int(index_node.text)

            if compound_index not in compounds:

                logging.debug(
                    "A PureOrMixtureData entry depends on an "
                    "unsupported compound and has been ignored"
                )

                return None

            if compound_index in compound_indices:

                raise ValueError(
                    "A ThermoML:PureOrMixtureData states its dependency on the "
                    "same compound twice."
                )

            compound_indices.append(compound_index)

        return compound_indices

    @staticmethod
    def extract_property_definitions(node, namespace, parent_phases):
        """Extract those property definitions defined by a PureOrMixtureData
        node. The extracted definitions are not extracted, as these a defined
        elsewhere in the archive file.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        parent_phases: PropertyPhase
            The phases specified by the parent PureOrMixtureData node.

        Returns
        ----------
        dict of int and ThermoMLProperty
            The extracted property definitions with keys of their
            assigned indices.
        """

        property_nodes = node.findall("ThermoML:Property", namespace)
        properties = {}

        for property_node in property_nodes:

            property_definition = ThermoMLProperty.from_xml_node(
                property_node, namespace, parent_phases
            )

            if property_definition is None:
                continue

            if property_definition.index in properties:

                raise ValueError(
                    "A ThermoML data set contains two properties with the same index"
                )

            properties[property_definition.index] = property_definition

        return properties

    @staticmethod
    def validate_constraint(constraint, compounds):
        """Validates a constraint object - this may be either
        a full `_Constraint` or just a `_VariableDefinition`.

        Parameters
        ----------
        constraint: _Constraint or _VariableDefinition
            The constraint to validate.
        compounds: dict of int and _Compound
            A dictionary of the compounds the parent PureOrMixtureData was
            measured for.

        Returns
        -------
        bool
            True if the constraint is valid, False otherwise.
        """

        if constraint is None or constraint.type is _ConstraintType.Undefined:

            logging.debug("An unsupported constraint has been ignored.")
            return False

        if (
            constraint.compound_index is not None
            and constraint.compound_index not in compounds
        ):

            logging.debug(
                "A constraint exists upon a non-existent compound and will be ignored."
            )
            return False

        if (
            constraint.type.is_composition_constraint()
            and constraint.compound_index is None
        ):

            logging.debug(
                "An unsupported constraint has been ignored - composition constraints"
                "need to have a corresponding compound_index."
            )
            return False

        return True

    @staticmethod
    def extract_global_constraints(node, namespace, compounds):
        """Extract the constraints which should be applied to all of
        the properties defined in a `PureOrMixtureData` node.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        compounds: dict of int and _Compound
            A dictionary of the compounds this PureOrMixtureData was
            measured for.

        Returns
        ----------
        list of _Constraint, optional
            The extracted constraints if all could be parsed,
            otherwise `None`.
        """

        constraint_nodes = node.findall("ThermoML:Constraint", namespace)
        constraints = []

        for constraint_node in constraint_nodes:

            constraint = _Constraint.from_node(constraint_node, namespace)

            if not _PureOrMixtureData.validate_constraint(constraint, compounds):
                return None

            constraints.append(constraint)

        return constraints

    @staticmethod
    def extract_variable_definitions(node, namespace, compounds):
        """Extract all of the 'variables' in a PureOrMixtureData node.
        These are simply constraints whose values are defined elsewhere
        in the archive.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        compounds: dict of int and _Compound
            A dictionary of the compounds this PureOrMixtureData was calculated for.

        Returns
        ----------
        dict of int and _VariableDefinition
            The extracted variable definitions which could be parsed.
        """
        variable_nodes = node.findall("ThermoML:Variable", namespace)
        variables = {}

        for variable_node in variable_nodes:

            variable = _VariableDefinition.from_node(variable_node, namespace)

            if not _PureOrMixtureData.validate_constraint(variable, compounds):
                continue

            variables[variable.index] = variable

        return variables

    @staticmethod
    def extract_uncertainty(node, namespace, property_definition):
        """Extracts the uncertainties on the measured properties
        contained in this `PureOrMixtureData` node.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        property_definition: ThermoMLProperty
            The property to which this uncertainty is attached.

        Returns
        -------
        float, optional
            The uncertainty in the property if it can be parsed or
            if one is present, otherwise `None`.
        """

        # Look for a standard uncertainty..
        uncertainty_node = node.find(".//ThermoML:nCombStdUncertValue", namespace)

        if uncertainty_node is None:
            uncertainty_node = node.find(".//ThermoML:nStdUncertValue", namespace)

        # We have found a std. uncertainty
        if uncertainty_node is not None:
            return float(uncertainty_node.text)

        # Try to calculate uncertainty from a coverage factor if present
        if (
            len(property_definition.combined_uncertainty_definitions) == 0
            and len(property_definition.property_uncertainty_definitions) == 0
        ):

            return None

        combined = len(property_definition.combined_uncertainty_definitions) > 0

        prefix = (
            _CombinedUncertainty.prefix if combined else _PropertyUncertainty.prefix
        )

        if combined:
            index_node = node.find(
                "./ThermoML:CombinedUncertainty/ThermoML:nCombUncertAssessNum",
                namespace,
            )
        else:
            index_node = node.find(
                "./ThermoML:PropUncertainty/ThermoML:nUncertAssessNum", namespace
            )

        expanded_uncertainty_node = node.find(
            ".//ThermoML:n" + prefix + "ExpandUncertValue", namespace
        )

        if index_node is None or expanded_uncertainty_node is None:
            return None

        expanded_uncertainty = float(expanded_uncertainty_node.text)
        index = int(index_node.text)

        if (
            combined
            and index not in property_definition.combined_uncertainty_definitions
        ):
            return None

        if (
            not combined
            and index not in property_definition.property_uncertainty_definitions
        ):
            return None

        divisor = (
            property_definition.combined_uncertainty_definitions[index].coverage_factor
            if combined
            else property_definition.property_uncertainty_definitions[
                index
            ].coverage_factor
        )

        return expanded_uncertainty / divisor

    @staticmethod
    def _smiles_to_molecular_weight(smiles):
        """Calculates the molecular weight of a substance specified
        by a smiles string.

        Parameters
        ----------
        smiles: str
            The smiles string to calculate the molecular weight of.

        Returns
        -------
        openff.evaluator.unit.Quantity
            The molecular weight.
        """

        from openff.toolkit.topology import Molecule

        try:
            from openmm import unit as openmm_unit
        except ImportError:
            from simtk.openmm import unit as openmm_unit

        try:

            molecule = Molecule.from_smiles(smiles)

        except Exception as e:

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)

            raise ValueError(
                f"The toolkit raised an exception for the "
                f"{smiles} smiles pattern: {formatted_exception}"
            )

        molecular_weight = 0.0 * openmm_unit.dalton

        for atom in molecule.atoms:
            molecular_weight += atom.mass

        return openmm_quantity_to_pint(molecular_weight)

    @staticmethod
    def _solvent_mole_fractions_to_moles(
        solvent_mass, solvent_mole_fractions, solvent_compounds
    ):
        """Converts a set of solvent mole fractions to moles for a
        given mass of solvent.

        Parameters
        ----------
        solvent_mass: openff.evaluator.unit.Quantity
            The total mass of the solvent in units compatible with kg.
        solvent_mole_fractions: dict of int and float
            The mole fractions of any solvent compounds in the system.
        solvent_compounds: dict of int and float
            A dictionary of any solvent compounds in the system.

        Returns
        -------
        dict of int and openff.evaluator.unit.Quantity
            A dictionary of the moles of each solvent compound.
        """
        weighted_molecular_weights = 0.0 * unit.gram / unit.mole
        number_of_moles = {}

        for solvent_index in solvent_compounds:

            solvent_smiles = solvent_compounds[solvent_index].smiles

            solvent_fraction = solvent_mole_fractions[solvent_index]
            solvent_weight = _PureOrMixtureData._smiles_to_molecular_weight(
                solvent_smiles
            )

            weighted_molecular_weights += solvent_weight * solvent_fraction

        total_solvent_moles = solvent_mass / weighted_molecular_weights

        for solvent_index in solvent_compounds:

            moles = solvent_mole_fractions[solvent_index] * total_solvent_moles
            number_of_moles[solvent_index] = moles

        return number_of_moles

    @staticmethod
    def _convert_mole_fractions(constraints, compounds, solvent_mole_fractions=None):
        """Converts a set of `_Constraint` to mole fractions.

        Parameters
        ----------
        constraints: list of _Constraint
            The constraints to convert.
        compounds: dict of int and _Compound
            The compounds in the system.
        solvent_mole_fractions: dict of int and float
            The mole fractions of any solvent compounds in the system,
            where the total mole fraction of all solvents must be equal
            to one.

        Returns
        -------
        dict of int and float
            A dictionary of compound indices and mole fractions.
        """

        # noinspection PyTypeChecker
        number_of_constraints = len(constraints)

        mole_fractions = {}
        total_mol_fraction = 0.0

        for constraint in constraints:

            mole_fraction = constraint.value

            if isinstance(mole_fraction, unit.Quantity):
                mole_fraction = mole_fraction.to(unit.dimensionless).magnitude

            mole_fractions[constraint.compound_index] = mole_fraction
            total_mol_fraction += mole_fractions[constraint.compound_index]

        if (
            number_of_constraints != len(compounds)
            and solvent_mole_fractions is not None
        ) or (
            number_of_constraints != len(compounds) - 1
            and number_of_constraints != len(compounds)
            and solvent_mole_fractions is None
        ):

            raise ValueError(
                f"The number of mole fraction constraints ({number_of_constraints}) must be one "
                f"less than or equal to the number of compounds being constrained ({len(compounds)}) "
                f"if a solvent list is not present, otherwise there must be an equal number."
            )

        # Handle the case were a single mole fraction constraint is missing.
        if number_of_constraints == len(compounds) - 1:

            for compound_index in compounds:

                if compound_index in mole_fractions:
                    continue

                mole_fractions[compound_index] = 1.0 - total_mol_fraction

        # Recompute the total mole fraction to be safe.
        total_mol_fraction = 0.0

        for compound_index in mole_fractions:
            total_mol_fraction += mole_fractions[compound_index]

        # Account for any solvent present.
        if solvent_mole_fractions is not None:

            # Assume the remainder of the mole fraction is the solvent.
            remaining_mole_fraction = 1.0 - total_mol_fraction

            for solvent_index in solvent_mole_fractions:
                mole_fractions[solvent_index] = (
                    solvent_mole_fractions[solvent_index] * remaining_mole_fraction
                )

        return mole_fractions

    @staticmethod
    def _convert_mass_fractions(
        constraints, compounds, solvent_mole_fractions=None, solvent_compounds=None
    ):
        """Converts a set of `_Constraint` to mole fractions.

        Parameters
        ----------
        constraints: list of _Constraint
            The constraints to convert.
        compounds: dict of int and _Compound
            The compounds in the system.
        solvent_mole_fractions: dict of int and float
            The mole fractions of any solvent compounds in the system,
            where the total mole fraction of all solvents must be equal
            to one.
        solvent_compounds: dict of int and float
            A dictionary of any explicitly defined solvent compounds in the
            system.

        Returns
        -------
        dict of int and float
            A dictionary of compound indices and mole fractions.
        """

        # noinspection PyTypeChecker
        number_of_constraints = len(constraints)

        mass_fractions = {}
        total_mass_fraction = 0.0

        for constraint in constraints:

            mass_fraction = constraint.value

            if isinstance(mass_fraction, unit.Quantity):
                mass_fraction = mass_fraction.to(unit.dimensionless).magnitude

            mass_fractions[constraint.compound_index] = mass_fraction
            total_mass_fraction += mass_fraction

        if (
            number_of_constraints != len(compounds)
            and solvent_mole_fractions is not None
        ) or (
            number_of_constraints != len(compounds) - 1
            and number_of_constraints != len(compounds)
            and solvent_mole_fractions is None
        ):

            raise ValueError(
                f"The number of mass fraction constraints ({number_of_constraints}) must be one "
                f"less than or equal to the number of compounds being constrained ({len(compounds)}) "
                f"if a solvent list is not present, otherwise there must be an equal number."
            )

        # Handle the case were a single mass fraction constraint is missing.
        if number_of_constraints == len(compounds) - 1:

            for compound_index in compounds:

                if compound_index in mass_fractions:
                    continue

                mass_fractions[compound_index] = 1.0 - total_mass_fraction
                if isinstance(mass_fractions[compound_index], unit.Quantity):
                    mass_fractions[compound_index] = (
                        mass_fractions[compound_index].to(unit.dimensionless).magnitude
                    )

        total_mass = 1 * unit.gram
        total_solvent_mass = total_mass

        moles = {}
        total_moles = 0.0 * unit.mole

        for compound_index in compounds:

            compound_smiles = compounds[compound_index].smiles
            compound_weight = _PureOrMixtureData._smiles_to_molecular_weight(
                compound_smiles
            )

            moles[compound_index] = (
                total_mass * mass_fractions[compound_index] / compound_weight
            )
            total_moles += moles[compound_index]

            total_solvent_mass -= total_mass * mass_fractions[compound_index]

        if (
            number_of_constraints == len(compounds)
            and solvent_mole_fractions is not None
        ):

            solvent_moles = _PureOrMixtureData._solvent_mole_fractions_to_moles(
                total_solvent_mass, solvent_mole_fractions, solvent_compounds
            )

            for solvent_index in solvent_moles:

                moles[solvent_index] = solvent_moles[solvent_index]
                total_moles += solvent_moles[solvent_index]

        mole_fractions = {}

        for compound_index in moles:

            mole_fraction = moles[compound_index] / total_moles
            mole_fractions[compound_index] = mole_fraction

        return mole_fractions

    @staticmethod
    def _convert_molality(
        constraints, compounds, solvent_mole_fractions=None, solvent_compounds=None
    ):
        """Converts a set of `_Constraint` to mole fractions.

        Parameters
        ----------
        constraints: list of _Constraint
            The constraints to convert.
        compounds: dict of int and _Compound
            The compounds in the system.
        solvent_mole_fractions: dict of int and float
            The mole fractions of any solvent compounds in the system,
            where the total mole fraction of all solvents must be equal
            to one.
        solvent_compounds: dict of int and float
            A dictionary of any explicitly defined solvent compounds in the
            system.

        Returns
        -------
        dict of int and float
            A dictionary of compound indices and mole fractions.
        """
        number_of_moles = {}
        # noinspection PyTypeChecker
        number_of_constraints = len(constraints)

        mole_fractions = {}

        total_number_of_moles = 0.0 * unit.moles
        total_solvent_mass = 1.0 * unit.kilograms

        for constraint in constraints:

            molality = constraint.value
            moles = molality * total_solvent_mass

            number_of_moles[constraint.compound_index] = moles
            total_number_of_moles += moles

        if (
            number_of_constraints != len(compounds) - 1
            and solvent_mole_fractions is None
        ) or (
            number_of_constraints != len(compounds)
            and solvent_mole_fractions is not None
        ):

            raise ValueError(
                f"The number of molality constraints ({number_of_constraints}) must be one "
                f"less than the number of compounds being constrained ({len(compounds)}) if a "
                f"solvent list is not present, otherwise there must be an equal number."
            )

        if (
            number_of_constraints == len(compounds) - 1
            and solvent_mole_fractions is None
        ):

            # In this case, there is no explicit solvent entry and the last component
            # whose molality has not been constrained is considered to be the 'solvent'
            for compound_index in compounds:

                if compound_index in number_of_moles:
                    continue

                compound_smiles = compounds[compound_index].smiles
                compound_weight = _PureOrMixtureData._smiles_to_molecular_weight(
                    compound_smiles
                )

                moles = total_solvent_mass / compound_weight

                number_of_moles[compound_index] = moles
                total_number_of_moles += moles

        elif (
            number_of_constraints == len(compounds)
            and solvent_mole_fractions is not None
        ):

            solvent_moles = _PureOrMixtureData._solvent_mole_fractions_to_moles(
                total_solvent_mass, solvent_mole_fractions, solvent_compounds
            )

            for solvent_index in solvent_moles:

                number_of_moles[solvent_index] = solvent_moles[solvent_index]
                total_number_of_moles += solvent_moles[solvent_index]

        for compound_index in number_of_moles:

            mole_fraction = number_of_moles[compound_index] / total_number_of_moles
            mole_fractions[compound_index] = mole_fraction

        return mole_fractions

    @staticmethod
    def build_substance(thermoml_property, constraints, compounds):
        """Build a Substance object from the extracted constraints and compounds.

        Parameters
        ----------
        thermoml_property: ThermoMLProperty
            The property to which this mixture belongs.
        constraints: list of _Constraint
            The ThermoML constraints.
        compounds: dict of int and _Compound
            A dictionary of the compounds this PureOrMixtureData was calculated for.

        Returns
        ----------
        Substance
            The constructed substance.
        """

        # TODO: We need to take into account `thermoml_property.target_compound_index` and
        #       `thermoml_property.solute_standard_state` to properly identify infinitely
        #       diluted solutes in the system (if any). Otherwise the solute will be
        #       assigned a mole fraction of zero.

        solvent_constraint_type = _ConstraintType.Undefined
        component_constraint_type = _ConstraintType.Undefined

        solvent_indices = set()

        for solvent_index in thermoml_property.solvents:

            if solvent_index in solvent_indices:
                continue

            solvent_indices.add(solvent_index)

        # Determine which types of solvent and component constraints are
        # being applied.
        for constraint in constraints:

            # Make sure we hunt down solvent indices.
            for solvent_index in constraint.solvents:

                if solvent_index in solvent_indices:
                    continue

                solvent_indices.add(solvent_index)

            # Only composition type restraints apply here, skip
            # the rest.
            if not constraint.type.is_composition_constraint():
                continue

            if (
                constraint.type == _ConstraintType.SolventMassFraction
                or constraint.type == _ConstraintType.SolventMoleFraction
                or constraint.type == _ConstraintType.SolventMolality
            ):

                if solvent_constraint_type == _ConstraintType.Undefined:
                    solvent_constraint_type = constraint.type

                if solvent_constraint_type != constraint.type:

                    logging.debug(
                        f"A property with different types of solvent composition constraints "
                        f"was found - {solvent_constraint_type} vs {constraint.type}). This "
                        f"is likely a bug in the ThermoML file and so this property will be "
                        f"skipped."
                    )

                    return None

            else:

                if component_constraint_type == _ConstraintType.Undefined:
                    component_constraint_type = constraint.type

                if component_constraint_type != constraint.type:

                    logging.debug(
                        f"A property with different types of composition constraints "
                        f"was found - {component_constraint_type} vs {constraint.type}). This "
                        f"is likely a bug in the ThermoML file and so this property will be "
                        f"skipped."
                    )

                    return None

        # If no constraint was applied, this very likely means a pure substance
        # was found.
        if (
            component_constraint_type == _ConstraintType.Undefined
            and solvent_constraint_type == _ConstraintType.Undefined
        ):

            component_constraint_type = _ConstraintType.ComponentMoleFraction

        elif (
            component_constraint_type == _ConstraintType.Undefined
            and solvent_constraint_type != _ConstraintType.Undefined
        ):

            logging.debug(
                f"A property with only solvent composition "
                f"constraints {solvent_constraint_type} was found."
            )

            return None

        solvent_mole_fractions = {}

        solvent_constraints = [
            constraint
            for constraint in constraints
            if constraint.type == solvent_constraint_type
        ]

        solvent_compounds = {}

        for solvent_index in solvent_indices:

            if solvent_index in compounds:

                solvent_compounds[solvent_index] = compounds[solvent_index]
                continue

            logging.debug(
                "The composition of a non-existent solvent was "
                "found. This usually only occurs in cases were "
                "the solvent component could not be understood "
                "by the framework."
            )

            return None

        # Make sure all of the solvents have not been removed.
        if (
            solvent_constraint_type != _ConstraintType.Undefined
            and len(solvent_indices) == 0
        ):

            logging.debug(
                "The composition of a solvent was found, however the "
                "solvent list is empty. This usually only occurs in "
                "cases were the solvent component could not be understood "
                "by the framework."
            )

            return None

        remaining_constraints = [
            constraint
            for constraint in constraints
            if constraint.type == component_constraint_type
        ]

        remaining_compounds = {}

        for compound_index in compounds:

            if compound_index in solvent_indices:
                continue

            remaining_compounds[compound_index] = compounds[compound_index]

        # Determine the mole fractions of the solvent species, if any.
        if solvent_constraint_type == _ConstraintType.SolventMoleFraction:

            solvent_mole_fractions = _PureOrMixtureData._convert_mole_fractions(
                solvent_constraints, solvent_compounds
            )

        elif solvent_constraint_type == _ConstraintType.SolventMassFraction:

            solvent_mole_fractions = _PureOrMixtureData._convert_mass_fractions(
                solvent_constraints, solvent_compounds
            )

        elif solvent_constraint_type == _ConstraintType.SolventMolality:

            solvent_mole_fractions = _PureOrMixtureData._convert_molality(
                solvent_constraints, solvent_compounds
            )

        elif solvent_constraint_type == _ConstraintType.Undefined:

            solvent_mole_fractions = None
            solvent_compounds = None

            remaining_compounds = compounds

        # Determine the mole fractions of the remaining compounds.
        mole_fractions = {}

        if component_constraint_type == _ConstraintType.ComponentMoleFraction:

            mole_fractions = _PureOrMixtureData._convert_mole_fractions(
                remaining_constraints, remaining_compounds, solvent_mole_fractions
            )

        elif component_constraint_type == _ConstraintType.ComponentMassFraction:

            mole_fractions = _PureOrMixtureData._convert_mass_fractions(
                remaining_constraints,
                remaining_compounds,
                solvent_mole_fractions,
                solvent_compounds,
            )

        elif component_constraint_type == _ConstraintType.ComponentMolality:

            mole_fractions = _PureOrMixtureData._convert_molality(
                remaining_constraints,
                remaining_compounds,
                solvent_mole_fractions,
                solvent_compounds,
            )

        if len(mole_fractions) != len(compounds):

            raise ValueError(
                f"The number of mole fractions ({len(mole_fractions)}) does not "
                f"equal the total number of compounds ({len(compounds)})"
            )

        # Make sure we haven't picked up a dimensionless unit be accident.
        for compound_index in mole_fractions:

            if isinstance(mole_fractions[compound_index], unit.Quantity):
                mole_fractions[compound_index] = (
                    mole_fractions[compound_index].to(unit.dimensionless).magnitude
                )

        total_mol_fraction = sum([value for value in mole_fractions.values()])

        if not np.isclose(total_mol_fraction, 1.0):
            raise ValueError(
                f"The total mole fraction {total_mol_fraction} is not equal to 1.0"
            )

        substance = Substance()

        for compound_index in compounds:

            compound = compounds[compound_index]

            if np.isclose(mole_fractions[compound_index], 0.0):
                continue

            substance.add_component(
                component=Component(smiles=compound.smiles),
                amount=MoleFraction(mole_fractions[compound_index]),
            )

        return substance

    @staticmethod
    def extract_measured_properties(
        node,
        namespace,
        property_definitions,
        global_constraints,
        variable_definitions,
        compounds,
    ):

        """Extract the measured properties defined by a ThermoML
        PureOrMixtureData node.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        property_definitions: dict of int and ThermoMLProperty
            The extracted property definitions.
        global_constraints: list of _Constraint
            The extracted constraints.
        variable_definitions: dict of int and _VariableDefinition
            The extracted variable definitions.
        compounds: dict of int and _Compound
            The extracted compounds.

        Returns
        ----------
        list of ThermoMLProperty
            The extracted measured properties.
        """

        value_nodes = node.findall("ThermoML:NumValues", namespace)

        measured_properties = []

        # Each value_node corresponds to one measure property.
        for value_node in value_nodes:

            constraints = []

            temperature_constraint = None
            pressure_constraint = None

            for global_constraint in global_constraints:

                constraint = copy.deepcopy(global_constraint)
                constraints.append(constraint)

                if constraint.type == _ConstraintType.Temperature:
                    temperature_constraint = constraint
                elif constraint.type == _ConstraintType.Pressure:
                    pressure_constraint = constraint

            # First extract the values of any variable constraints
            variable_nodes = value_node.findall("ThermoML:VariableValue", namespace)

            skip_entry = False

            for variable_node in variable_nodes:

                variable_index = int(
                    variable_node.find("./ThermoML:nVarNumber", namespace).text
                )

                if variable_index not in variable_definitions:

                    # The property was constrained by an unsupported variable and
                    # so will be skipped for now.
                    skip_entry = True
                    break

                variable_definition = variable_definitions[variable_index]

                variable_value = float(
                    variable_node.find("ThermoML:nVarValue", namespace).text
                )
                value_as_quantity = unit.Quantity(
                    variable_value, variable_definition.default_unit
                )

                # Convert the 'variable' into a full constraint entry
                constraint = _Constraint.from_variable(
                    variable_definition, value_as_quantity
                )
                constraints.append(constraint)

                if constraint.type == _ConstraintType.Temperature:
                    temperature_constraint = constraint
                elif constraint.type == _ConstraintType.Pressure:
                    pressure_constraint = constraint

            if skip_entry:
                continue

            # Extract the thermodynamic state that the property was measured at.
            if temperature_constraint is None:

                logging.debug(
                    "A property did not report the temperature it "
                    "was measured at and will be ignored."
                )
                continue

            temperature = temperature_constraint.value
            pressure = (
                None if pressure_constraint is None else pressure_constraint.value
            )

            thermodynamic_state = ThermodynamicState(
                temperature=temperature, pressure=pressure
            )

            # Now extract the actual values of the measured properties, and their
            # uncertainties
            property_nodes = value_node.findall("ThermoML:PropertyValue", namespace)

            for property_node in property_nodes:

                property_index = int(
                    property_node.find("./ThermoML:nPropNumber", namespace).text
                )

                if property_index not in property_definitions:

                    # Most likely the property was dropped earlier due to an unsupported phase / type
                    continue

                property_definition = property_definitions[property_index]

                uncertainty = _PureOrMixtureData.extract_uncertainty(
                    property_node, namespace, property_definition
                )

                measured_property = copy.deepcopy(property_definition)

                measured_property.thermodynamic_state = thermodynamic_state

                property_value_node = property_node.find(
                    ".//ThermoML:nPropValue", namespace
                )

                measured_property.set_value(
                    float(property_value_node.text),
                    None if uncertainty is None else float(uncertainty),
                )

                mixture = _PureOrMixtureData.build_substance(
                    measured_property, constraints, compounds
                )

                if mixture is None:
                    continue

                measured_property.substance = mixture
                measured_properties.append(measured_property)

        # By this point we now have the measured properties and the thermodynamic state
        # they were measured at converted to standardised classes.
        return measured_properties

    @staticmethod
    def from_xml_node(node, namespace, compounds):
        """Extracts all of the data in a ThermoML PureOrMixtureData node.

        Parameters
        ----------
        node: xml.etree.Element
            The xml node to read.
        namespace: dict of str and str
            The xml namespace.
        compounds: dict of int and _Compound
            A list of the already extracted `_Compound`'s.

        Returns
        ----------
        list of ThermoMLProperty
            A list of extracted properties.
        """

        # Figure out which compounds are going to be associated with
        # the property entries.
        compound_indices = _PureOrMixtureData.extract_compound_indices(
            node, namespace, compounds
        )

        if compound_indices is None:
            # Most likely this entry depended on a non-parsable compound
            # and will be skipped entirely
            return None

        if len(compound_indices) == 0:

            logging.debug("A PureOrMixtureData entry with no compounds was ignored.")
            return None

        phase_nodes = node.findall("./ThermoML:PhaseID/ThermoML:ePhase", namespace)

        all_phases = None

        for phase_node in phase_nodes:

            phase = _phase_from_thermoml_string(phase_node.text)

            if phase == PropertyPhase.Undefined:

                logging.debug(
                    f"A property was measured in an unsupported phase "
                    f"({phase_node.text}) and will be skipped."
                )

                return None

            all_phases = phase if all_phases is None else all_phases | phase

        # Extract property definitions - values come later!
        property_definitions = _PureOrMixtureData.extract_property_definitions(
            node, namespace, all_phases
        )

        if len(property_definitions) == 0:
            return None

        for property_index in property_definitions:

            all_phases |= property_definitions[property_index].phase
            property_definitions[property_index].phase |= all_phases

        # Extract any constraints on the system e.g pressure, temperature
        global_constraints = _PureOrMixtureData.extract_global_constraints(
            node, namespace, compounds
        )

        if global_constraints is None:
            return None

        # Extract any variables set on the system e.g pressure, temperature
        # Only the definition entry and not the value of the variable is extracted
        variable_definitions = _PureOrMixtureData.extract_variable_definitions(
            node, namespace, compounds
        )

        if len(global_constraints) == 0 and len(variable_definitions) == 0:

            logging.debug("A PureOrMixtureData entry with no constraints was ignored.")
            return None

        used_compounds = {}

        for compound_index in compounds:

            if compound_index not in compound_indices:
                continue

            used_compounds[compound_index] = compounds[compound_index]

        measured_properties = _PureOrMixtureData.extract_measured_properties(
            node,
            namespace,
            property_definitions,
            global_constraints,
            variable_definitions,
            used_compounds,
        )

        return measured_properties


class ThermoMLProperty:
    """A wrapper around a ThermoML Property node."""

    class SoluteStandardState(Enum):
        """Describes the standard state of a solute."""

        Undefined = ("Undefined",)
        InfiniteDilutionSolute = ("Infinite dilution solute",)
        PureCompound = ("Pure compound",)
        PureLiquidSolute = ("Pure liquid solute",)
        StandardMolality = ("Standard molality (1 mol/kg) solute",)

        @staticmethod
        def from_node(node):
            """Converts an `eStandardState` node a `ThermoMLProperty.SoluteStandardState`.

            Parameters
            ----------
            node: xml.etree.Element
                The xml node to convert.

            Returns
            ----------
            ThermoMLProperty.SoluteStandardState
                The converted state type.
            """

            try:
                standard_state = ThermoMLProperty.SoluteStandardState(node.text)
            except (KeyError, ValueError):
                standard_state = ThermoMLProperty.SoluteStandardState.Undefined

            if standard_state == _ConstraintType.Undefined:

                logging.debug(
                    f"{node.tag}->{node.text} is an unsupported "
                    f"solute standard state type."
                )

            return standard_state

    def __init__(self, type_string):

        self.type_string = type_string

        self.thermodynamic_state = None
        self.phase = PropertyPhase.Undefined

        self.substance = None

        self.value = None
        self.uncertainty = None

        self.source = None

        self.index = None

        self.solute_standard_state = ThermoMLProperty.SoluteStandardState.Undefined
        self.solvents = []

        self.target_compound_index = None

        self.property_uncertainty_definitions = {}
        self.combined_uncertainty_definitions = {}

        self.default_unit = None

        self.target_compound_index = None

    @staticmethod
    def extract_uncertainty_definitions(
        node,
        namespace,
        property_uncertainty_definitions,
        combined_uncertainty_definitions,
    ):

        """Extract any property or combined uncertainties from a property xml node.

        Parameters
        ----------
        node: Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.
        property_uncertainty_definitions: list(_PropertyUncertainty)
            A list of the extracted property uncertainties.
        combined_uncertainty_definitions: list(_PropertyUncertainty)
            A list of the extracted combined property uncertainties.
        """

        property_nodes = node.findall("ThermoML:CombinedUncertainty", namespace)

        for property_node in property_nodes:

            if property_node is None:
                continue

            uncertainty_definition = _CombinedUncertainty.from_xml(
                property_node, namespace
            )

            if uncertainty_definition is None:
                continue

            combined_uncertainty_definitions[
                uncertainty_definition.index
            ] = uncertainty_definition

        property_nodes = node.findall("ThermoML:PropUncertainty", namespace)

        for property_node in property_nodes:

            if property_node is None:
                continue

            uncertainty_definition = _PropertyUncertainty.from_xml(
                property_node, namespace
            )

            if uncertainty_definition is None:
                continue

            property_uncertainty_definitions[
                uncertainty_definition.index
            ] = uncertainty_definition

    @classmethod
    def from_xml_node(cls, node, namespace, parent_phases):
        """Creates a ThermoMLProperty from an xml node.

        Parameters
        ----------
        node: Element
            The xml node to convert.
        namespace: dict of str and str
            The xml namespace.
        parent_phases: PropertyPhase
            The phases specfied in the parent PureOrMixtureData node.

        Returns
        ----------
        _Compound
            The created property.
        """

        # Gather up all possible identifiers
        index_node = node.find("ThermoML:nPropNumber", namespace)

        property_index = int(index_node.text)

        phase_node = node.find("./ThermoML:PropPhaseID//ThermoML:ePropPhase", namespace)
        phase = PropertyPhase.Undefined | parent_phases

        if phase_node is not None:
            phase |= _phase_from_thermoml_string(phase_node.text)

        reference_phase_node = node.find(
            "./ThermoML:RefPhaseID//ThermoML:eRefPhase", namespace
        )

        if reference_phase_node is not None:
            phase |= _phase_from_thermoml_string(reference_phase_node.text)

        if phase == PropertyPhase.Undefined:

            logging.debug(
                f"A property was measured in an unsupported phase "
                f"({phase_node.text}) and will be skipped."
            )

            return None

        property_group_node = node.find(
            "./ThermoML:Property-MethodID//ThermoML:PropertyGroup//*", namespace
        )

        property_name_node = property_group_node.find("./ThermoML:ePropName", namespace)
        method_name_node = property_group_node.find("./ThermoML:eMethodName", namespace)

        if method_name_node is None:
            method_name_node = property_group_node.find(
                "./ThermoML:sMethodName", namespace
            )

        if method_name_node is None or property_name_node is None:
            raise RuntimeError("A property does not have a name / method entry.")

        if property_name_node.text not in ThermoMLDataSet.registered_properties:

            logging.debug(
                f"An unsupported property was found "
                f"({property_name_node.text}) and will be skipped."
            )

            return None

        registered_plugin = ThermoMLDataSet.registered_properties[
            property_name_node.text
        ]

        if (registered_plugin.supported_phases & phase) != phase:

            logging.debug(
                f"The {property_name_node.text} property is currently only supported "
                f"when measured in the {str(registered_plugin.supported_phases)} phase, "
                f"and not the {str(phase)} phase."
            )

            return None

        return_value = cls(property_name_node.text)

        return_value.index = property_index
        return_value.phase = phase

        return_value.default_unit = _unit_from_thermoml_string(property_name_node.text)

        return_value.method_name = method_name_node.text

        property_uncertainty_definitions = {}
        combined_uncertainty_definitions = {}

        cls.extract_uncertainty_definitions(
            node,
            namespace,
            property_uncertainty_definitions,
            combined_uncertainty_definitions,
        )

        return_value.combined_uncertainty_definitions = combined_uncertainty_definitions
        return_value.property_uncertainty_definitions = property_uncertainty_definitions

        solvent_index_nodes = node.findall(
            "./ThermoML:Solvent//ThermoML:nOrgNum", namespace
        )

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node.text))

        # The solute standard state appears to describe which a solute should
        # be present in only trace amounts. It only seems to be relevant for
        # activity based properties.
        standard_state_node = node.find("./ThermoML:eStandardState", namespace)

        if standard_state_node is not None:
            return_value.solute_standard_state = (
                ThermoMLProperty.SoluteStandardState.from_node(standard_state_node)
            )

        # Property->Property-MethodID->RegNum describes which compound is referred
        # to if the property is based on one of the compounds e.g. the activity
        # coefficient of compound 2.
        target_compound_node = node.find(
            "./ThermoML:Property-MethodID/ThermoML:RegNum/ThermoML:nOrgNum", namespace
        )

        if target_compound_node is not None:
            return_value.target_compound_index = int(target_compound_node.text)

        return return_value

    def set_value(self, value, uncertainty):
        """Set the value and uncertainty of this property, adding units if necessary.

        Parameters
        ----------
        value: float or unit.Quantity
            The value of the property
        uncertainty: float or unit.Quantity, optional
            The uncertainty in the value.
        """
        value_quantity = value

        if not isinstance(value_quantity, unit.Quantity):
            value_quantity = value * self.default_unit

        self.value = value_quantity

        if uncertainty is not None:

            uncertainty_quantity = uncertainty

            if not isinstance(uncertainty_quantity, unit.Quantity):
                uncertainty_quantity = uncertainty * self.default_unit

            self.uncertainty = uncertainty_quantity


class ThermoMLDataSet(PhysicalPropertyDataSet):
    """A dataset of physical property measurements created from a ThermoML dataset.

    Examples
    --------

    For example, we can use the DOI `10.1016/j.jct.2005.03.012` as a key
    for retrieving the dataset from the ThermoML Archive:

    >>> dataset = ThermoMLDataSet.from_doi('10.1016/j.jct.2005.03.012')

    You can also specify multiple ThermoML Archive keys to create a dataset from multiple ThermoML files:

    >>> thermoml_keys = ['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474']
    >>> dataset = ThermoMLDataSet.from_doi(*thermoml_keys)

    """

    registered_properties = {}

    def __init__(self):
        """Constructs a new ThermoMLDataSet object."""
        super().__init__()

    @classmethod
    def from_doi(cls, *doi_list):
        """Load a ThermoML data set from a list of DOIs

        Parameters
        ----------
        doi_list: str
            The list of DOIs to pull data from

        Returns
        -------
        ThermoMLDataSet
            The loaded data set.
        """
        return_value = None

        for doi in doi_list:

            # E.g https://trc.nist.gov/ThermoML/10.1016/j.jct.2016.12.009.xml
            doi_url = f"https://trc.nist.gov/ThermoML/{doi}.xml"

            data_set = cls._from_url(doi_url, MeasurementSource(doi=doi))

            if data_set is None or len(data_set) == 0:
                continue

            if return_value is None:
                return_value = data_set
            else:
                return_value.merge(data_set)

        return return_value

    @classmethod
    def from_url(cls, *url_list):
        """Load a ThermoML data set from a list of URLs

        Parameters
        ----------
        url_list: str
            The list of URLs to pull data from

        Returns
        -------
        ThermoMLDataSet
            The loaded data set.
        """

        return_value = None

        for url in url_list:

            data_set = cls._from_url(url)

            if data_set is None or len(data_set) == 0:
                continue

            if return_value is None:
                return_value = data_set
            else:
                return_value.merge(data_set)

        return return_value

    @classmethod
    def _from_url(cls, url, source=None):
        """Load a ThermoML data set from a given URL

        Parameters
        ----------
        url: str
            The URL to pull data from
        source: Source, optional
            An optional source which gives more information (e.g DOIs) for the url.

        Returns
        ----------
        ThermoMLDataSet
            The loaded data set.
        """
        if source is None:
            source = MeasurementSource(reference=url)

        return_value = None

        try:

            request = requests.get(url)
            request.raise_for_status()

            # Handle the case where ThermoML returns a 404 error code, but rather
            # redirects to an error page with code 200.
            if request.text.startswith("<html>"):
                raise HTTPError(url, 404, "Not found", None, None)

            return_value = cls.from_xml(request.text, source)

        except (HTTPError, requests.exceptions.HTTPError):
            logger.warning(f"No ThermoML file could not be found at {url}")

        return return_value

    @classmethod
    def from_file(cls, *file_list):
        """Load a ThermoML data set from a list of files

        Parameters
        ----------
        file_list: str
            The list of files to pull data from

        Returns
        -------
        ThermoMLDataSet
            The loaded data set.
        """
        return_value = None
        counter = 0

        for file in file_list:

            data_set = cls._from_file(file)

            counter += 1

            if data_set is None or len(data_set) == 0:
                continue

            if return_value is None:
                return_value = data_set
            else:
                return_value.merge(data_set)

        return return_value

    @classmethod
    def _from_file(cls, path):
        """Load a ThermoML data set from a given file

        Parameters
        ----------
        path: str
            The file path to pull data from

        Returns
        -------
        ThermoMLDataSet
            The loaded data set.
        """
        source = MeasurementSource(reference=path)
        return_value = None

        try:

            with open(path) as file:
                return_value = ThermoMLDataSet.from_xml(file.read(), source)

        except FileNotFoundError:
            logger.warning(f"No ThermoML file could not be found at {path}")

        return return_value

    @classmethod
    def from_xml(cls, xml, default_source):
        """Load a ThermoML data set from an xml object.

        Parameters
        ----------
        xml: str
            The xml string to parse.
        default_source: Source
            The source to use if one cannot be parsed from the archive itself.

        Returns
        -------
        ThermoMLDataSet
            The loaded ThermoML data set.
        """
        root_node = ElementTree.fromstring(xml)

        if root_node is None:
            logger.warning("The ThermoML XML document could not be parsed.")
            return None

        if root_node.tag.find("DataReport") < 0:
            logger.warning(
                "The ThermoML XML document does not contain the expected root node."
            )
            return None

        # Extract the namespace that will prefix all type names
        namespace_string = re.search(r"{.*\}", root_node.tag).group(0)[1:-1]
        namespace = {"ThermoML": namespace_string}

        # Attempt to find a DOI for this archive
        doi_node = root_node.find("ThermoML:Citation/ThermoML:sDOI", namespace)

        if doi_node is not None:
            source = MeasurementSource(doi=doi_node.text)
        else:
            source = default_source

        return_value = ThermoMLDataSet()
        compounds = {}

        # Extract the base compounds present in the xml file
        for node in root_node.findall("ThermoML:Compound", namespace):

            compound = _Compound.from_xml_node(node, namespace)

            if compound is None:
                continue

            if compound.index in compounds:
                raise RuntimeError(
                    "A ThermoML data set contains two compounds with the same index"
                )

            compounds[compound.index] = compound

        # Pull out any and all properties in the file.
        for node in root_node.findall("ThermoML:PureOrMixtureData", namespace):

            properties = _PureOrMixtureData.from_xml_node(node, namespace, compounds)

            if properties is None or len(properties) == 0:
                continue

            for measured_property in properties:

                registered_plugin = ThermoMLDataSet.registered_properties[
                    measured_property.type_string
                ]

                mapped_property = registered_plugin.conversion_function(
                    measured_property
                )
                mapped_property.source = source
                return_value.add_properties(mapped_property)

        return return_value
