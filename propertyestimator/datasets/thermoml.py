"""
An API for importing a ThermoML archive.
"""

import logging
import pickle
import re
from enum import unique, Enum
from urllib.error import HTTPError
from urllib.request import urlopen
from xml.etree import ElementTree

import numpy as np
from simtk import unit

from propertyestimator.properties import PropertyPhase, MeasurementSource
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from .datasets import PhysicalPropertyDataSet
from .plugins import registered_thermoml_properties


def unit_from_thermoml_string(full_string):
    """A non-ideal way to convert a string to a simtk.unit.Unit

    Parameters
    ----------
    full_string : str
        The string to convert to a simtk Unit

    Returns
    ----------
    simtk.unit.Unit, None
        None if the string is unitless, otherwise a simtk.unit.Unit
    """

    full_string_split = full_string.split(',')

    unit_string = full_string_split[1] if len(full_string_split) > 1 else ''
    unit_string = unit_string.strip()

    if unit_string == 'K':
        return unit.kelvin
    elif unit_string == 'kPa':
        return unit.kilo * unit.pascal
    elif unit_string == 'kg/m3':
        return unit.kilogram / unit.meter**3
    elif unit_string == 'mol/kg':
        return unit.mole / unit.kilogram
    elif unit_string == 'mol/dm3':
        return unit.mole / unit.decimeter ** 3
    elif unit_string == 'kJ/mol':
        return unit.kilojoule_per_mole
    elif unit_string == 'm3/kg':
        return unit.meter ** 3 / unit.kilogram
    elif unit_string == 'mol/m3':
        return unit.mole / unit.meter ** 3
    elif unit_string == 'm3/mol':
        return unit.meter**3 / unit.mole
    elif unit_string == 'J/K/mol':
        return unit.joule / unit.kelvin / unit.mole
    elif unit_string == 'J/K/kg':
        return unit.joule / unit.kelvin / unit.kilogram
    elif unit_string == 'J/K/m3':
        return unit.joule / unit.kelvin / unit.meter ** 3
    elif unit_string == '1/kPa':
        return (1.0 / unit.kilopascal).unit
    elif unit_string == 'm/s':
        return unit.meter / unit.second
    elif unit_string == 'MHz':
        return (1.0 / unit.megasecond).unit
    elif unit_string == 'N/m':
        return unit.newton / unit.meter
    elif len(unit_string) == 0:
        return None
    else:
        raise NotImplementedError('The unit (' + unit_string + ') is not currently supported')


def phase_from_thermoml_string(string):
    """Converts a ThermoML string to a PropertyPhase

        Parameters
        ----------
        string : str
            The string to convert to a PropertyPhase

        Returns
        ----------
        PropertyPhase
            The converted PropertyPhase
        """
    phase_string = string.lower().strip()
    phase = PropertyPhase.Undefined

    if (phase_string == 'liquid' or
        phase_string.find('fluid') >= 0 or phase_string.find('solution') >= 0):

        phase = PropertyPhase.Liquid

    elif phase_string.find('crystal') >= 0 and not phase_string.find('liquid') >= 0:

        phase = PropertyPhase.Solid

    elif phase_string.find('gas') >= 0:

        phase = PropertyPhase.Gas

    return phase


@unique
class ThermoMLConstraintType(Enum):
    """An enum containing the supported types of ThermoML constraints
    """

    Undefined = 'Undefined'
    Temperature = 'Temperature, K'
    Pressure = 'Pressure, kPa'
    ComponentMoleFraction = 'Mole fraction'
    ComponentMassFraction = 'Mass fraction'
    ComponentMolality = 'Molality, mol/kg'
    SolventMoleFraction = 'Solvent: Mole fraction'
    SolventMassFraction = 'Solvent: Mass fraction'
    SolventMolality = 'Solvent: Molality, mol/kg'

    @staticmethod
    def from_node(node):
        """Converts either a ConstraintType or VariableType xml node to a ThermoMLConstraintType.

        Parameters
        ----------
        node : xml.etree.Element
            The xml node to convert.

        Returns
        ----------
        ThermoMLConstraintType
            The converted constraint type.
        """

        try:
            constraint_type = ThermoMLConstraintType(node.text)
        except (KeyError, ValueError):
            constraint_type = ThermoMLConstraintType.Undefined

        if constraint_type == ThermoMLConstraintType.Undefined:
            logging.warning(node.tag + '->' + node.text + ' is an unsupported constraint type.')

        return constraint_type

    def is_composition_constraint(self):
        """Checks whether the purpose of this constraint is
        to constrain the substance composition.

        Returns
        -------
        bool
            True if the constraint type is either a

            - `ThermoMLConstraintType.ComponentMoleFraction`
            - `ThermoMLConstraintType.ComponentMassFraction`
            - `ThermoMLConstraintType.ComponentMolality`
            - `ThermoMLConstraintType.SolventMoleFraction`
            - `ThermoMLConstraintType.SolventMassFraction`
            - `ThermoMLConstraintType.SolventMolality`
        """
        return (self == ThermoMLConstraintType.ComponentMoleFraction or
                self == ThermoMLConstraintType.ComponentMassFraction or
                self == ThermoMLConstraintType.ComponentMolality or
                self == ThermoMLConstraintType.SolventMoleFraction or
                self == ThermoMLConstraintType.SolventMassFraction or
                self == ThermoMLConstraintType.SolventMolality)


class ThermoMLConstraint:
    """A wrapper around a ThermoML Constraint node.
    """
    def __init__(self):

        self.type = ThermoMLConstraintType.Undefined
        self.value = 0.0

        self.solvents = []

        # Describes which compound the variable acts upon.
        self.compound_index = None

    @classmethod
    def from_node(cls, constraint_node, namespace):
        """Creates a ThermoMLConstraint from an xml node.

        Parameters
        ----------
        constraint_node : Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        ThermoMLConstraint
            The created constraint.
        """
        # Extract the xml nodes.
        type_node = constraint_node.find('.//ThermoML:ConstraintType/*', namespace)
        value_node = constraint_node.find('./ThermoML:nConstraintValue', namespace)

        solvent_index_nodes = constraint_node.find('./ThermoML:Solvent//nOrgNum', namespace)
        compound_index_node = constraint_node.find('./ThermoML:ConstraintID/ThermoML:RegNum/*', namespace)

        value = float(value_node.text)
        # Determine what the default unit for this variable should be.
        unit_type = unit_from_thermoml_string(type_node.text)

        return_value = cls()

        return_value.type = ThermoMLConstraintType.from_node(type_node)
        return_value.value = unit.Quantity(value, unit_type)

        if compound_index_node is not None:
            return_value.compound_index = int(compound_index_node.text)

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node))

        return None if return_value.type is ThermoMLConstraintType.Undefined else return_value

    @classmethod
    def from_variable(cls, variable, value):
        """Creates a ThermoMLConstraint from an existing ThermoML variable definition.

        Parameters
        ----------
        variable : ThermoMLVariableDefinition
            The variable to convert.
        value : simtk.unit.Quantity
            The value of the constant.

        Returns
        ----------
        ThermoMLConstraint
            The created constraint.
        """
        return_value = cls()

        return_value.type = variable.type
        return_value.compound_index = variable.compound_index

        return_value.solvents.extend(variable.solvents)

        return_value.value = value

        return return_value


class ThermoMLVariableDefinition:
    """A wrapper around a ThermoML Variable node.
    """
    def __init__(self):

        self.index = -1
        self.type = ThermoMLConstraintType.Undefined

        self.solvents = []
        self.default_unit = None

        # Describes which compound the variable acts upon.
        self.compound_index = None

    @classmethod
    def from_node(cls, variable_node, namespace):
        """Creates a ThermoMLVariableDefinition from an xml node.

        Parameters
        ----------
        variable_node : xml.etree.Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        ThermoMLVariableDefinition
            The created variable definition.
        """
        # Extract the xml nodes.
        type_node = variable_node.find('.//ThermoML:VariableType/*', namespace)
        index_node = variable_node.find('ThermoML:nVarNumber', namespace)

        solvent_index_nodes = variable_node.find('./ThermoML:Solvent//nOrgNum', namespace)
        compound_index_node = variable_node.find('./ThermoML:VariableID/ThermoML:RegNum/*', namespace)

        return_value = cls()

        return_value.default_unit = unit_from_thermoml_string(type_node.text)

        return_value.index = int(index_node.text)
        return_value.type = ThermoMLConstraintType.from_node(type_node)

        if compound_index_node is not None:
            return_value.compound_index = int(compound_index_node.text)

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node))

        return None if return_value.type is ThermoMLConstraintType.Undefined else return_value


class ThermoMLPropertyUncertainty:
    """A wrapper around a ThermoML PropUncertainty node.
    """

    # Reduce code redundancy by reusing this class for
    # both property and combined uncertainties.
    prefix = ''

    def __init__(self):

        self.index = -1
        self.coverage_factor = None

    @classmethod
    def from_xml(cls, node, namespace):
        """Creates a ThermoMLPropertyUncertainty from an xml node.

        Parameters
        ----------
        node : Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        ThermoMLCompound
            The created property uncertainty.
        """

        coverage_factor_node = node.find('ThermoML:n' + cls.prefix + 'CoverageFactor', namespace)
        confidence_node = node.find('ThermoML:n' + cls.prefix + 'UncertLevOfConfid', namespace)

        # As defined by https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-7-reporting-uncertainty
        if coverage_factor_node is not None:
            coverage_factor = float(coverage_factor_node.text)
        elif confidence_node is not None and confidence_node.text == '95':
            coverage_factor = 2
        else:
            return None

        index_node = node.find('ThermoML:n' + cls.prefix + 'UncertAssessNum', namespace)
        index = int(index_node.text)

        return_value = cls()

        return_value.coverage_factor = coverage_factor
        return_value.index = index

        return return_value


class ThermoMLCombinedUncertainty(ThermoMLPropertyUncertainty):
    """A wrapper around a ThermoML CombPropUncertainty node.
    """

    prefix = 'Comb'


class ThermoMLCompound:
    """A wrapper around a ThermoML Compound node.
    """
    def __init__(self):

        self.smiles = None
        self.index = -1

    @staticmethod
    def smiles_from_inchi_string(inchi_string):
        """Attempts to create a SMILES pattern from an inchi string.

        Todo: SMILES from InChI should exist at the toolkit
              level in an OEChem independent way.

        Parameters
        ----------
        inchi_string : str
            The InChI string to convert.

        Returns
        ----------
        str, optional
            None if the identifier cannot be converted, otherwise the converted SMILES pattern.
        """
        from openeye import oechem

        if inchi_string is None:
            raise ValueError('The InChI string cannot be `None`.')

        temp_molecule = oechem.OEMol()

        if oechem.OEParseInChI(temp_molecule, inchi_string) is False:
            raise ValueError('All InChI strings in ThermoML files must be valid.')

        return oechem.OEMolToSmiles(temp_molecule)

    @classmethod
    def from_xml_node(cls, node, namespace):
        """Creates a ThermoMLCompound from an xml node.

        Parameters
        ----------
        node : Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        ThermoMLCompound
            The created compound wrapper.
        """
        # Gather up all possible identifiers
        identifier_nodes = node.findall('ThermoML:sStandardInChI', namespace)

        if len(identifier_nodes) == 0 or identifier_nodes[0].text is None:
            # convert common name to smiles
            raise ValueError('A ThermoML:Compound node does not have a valid InChI identifier')

        smiles = cls.smiles_from_inchi_string(identifier_nodes[0].text)

        index_node = node.find('./ThermoML:RegNum/*', namespace)

        if index_node is None:
            raise ValueError('A ThermoML:Compound has a non-existent index')

        compound_index = int(index_node.text)

        return_value = cls()

        return_value.smiles = smiles
        return_value.index = compound_index

        return return_value


class ThermoMLProperty:
    """A wrapper around a ThermoML Property node.
    """
    def __init__(self, base_type):

        self.thermodynamic_state = None
        self.phase = PropertyPhase.Undefined

        self.substance = None

        self.value = None
        self.uncertainty = None

        self.source = None

        self.index = None

        self.type = base_type

        self.solvents = []

        self.property_uncertainty_definitions = {}
        self.combined_uncertainty_definitions = {}

        self.default_unit = None

    @property
    def temperature(self):
        """simtk.unit.Quantity or None: The temperature at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.temperature

    @property
    def pressure(self):
        """simtk.unit.Quantity or None: The pressure at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.pressure

    @staticmethod
    def extract_uncertainty_definitions(node, namespace,
                                        property_uncertainty_definitions,
                                        combined_uncertainty_definitions):

        """Extract any property or combined uncertainties from a property xml node.

        Parameters
        ----------
        node : Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.
        property_uncertainty_definitions : list(ThermoMLPropertyUncertainty)
            A list of the extracted property uncertainties.
        combined_uncertainty_definitions : list(ThermoMLPropertyUncertainty)
            A list of the extracted combined property uncertainties.
        """

        property_nodes = node.findall('ThermoML:CombinedUncertainty', namespace)

        for property_node in property_nodes:

            if property_node is None:
                continue

            uncertainty_definition = ThermoMLCombinedUncertainty.from_xml(property_node, namespace)

            if uncertainty_definition is None:
                continue

            combined_uncertainty_definitions[uncertainty_definition.index] = uncertainty_definition

        property_nodes = node.findall('ThermoML:PropUncertainty', namespace)

        for property_node in property_nodes:

            if property_node is None:
                continue

            uncertainty_definition = ThermoMLPropertyUncertainty.from_xml(property_node, namespace)

            if uncertainty_definition is None:
                continue

            property_uncertainty_definitions[uncertainty_definition.index] = uncertainty_definition

    @classmethod
    def from_xml_node(cls, node, namespace):
        """Creates a ThermoMLProperty from an xml node.

        Parameters
        ----------
        node : Element
            The xml node to convert.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        ThermoMLCompound
            The created property.
        """

        # Gather up all possible identifiers
        index_node = node.find('ThermoML:nPropNumber', namespace)

        property_index = int(index_node.text)

        phase_node = node.find('./ThermoML:PropPhaseID//ThermoML:ePropPhase', namespace)
        phase = PropertyPhase.Undefined

        if phase_node is not None:
            phase = phase_from_thermoml_string(phase_node.text)

        if phase == PropertyPhase.Undefined:

            # TODO: For now we just hope that the property defines the phase.
            #       This needs to be better supported however.
            logging.warning('A property was measured in an unsupported phase (' +
                            phase_node.text + ') and will be skipped.')

            return None

        # TODO: Property->RegNum is currently ignored
        # Describes which compound is referred to if the property is based on one of
        # the compounds... e.g. mass fraction of compound 2

        property_group_node = node.find('./ThermoML:Property-MethodID//ThermoML:PropertyGroup//*', namespace)

        property_name_node = property_group_node.find('./ThermoML:ePropName', namespace)
        method_name_node = property_group_node.find('./ThermoML:eMethodName', namespace)

        if method_name_node is None:
            method_name_node = property_group_node.find('./ThermoML:sMethodName', namespace)

        if method_name_node is None or property_name_node is None:
            raise RuntimeError('A property does not have a name / method entry.')

        if property_name_node.text not in registered_thermoml_properties:

            logging.warning('An unsupported property was found ({}) and '
                            'will be skipped.'.format(property_name_node.text))

            return None

        registered_plugin = registered_thermoml_properties[property_name_node.text]

        if (registered_plugin.supported_phases & phase) != phase:

            logging.warning(f'The {property_name_node.text} property is currently only supported'
                            f'when measured for {phase} phase properties.')

            return None

        return_value = cls(registered_plugin.class_type)

        return_value.index = property_index
        return_value.phase = phase

        return_value.default_unit = unit_from_thermoml_string(property_name_node.text)

        return_value.type = registered_plugin.class_type
        return_value.method_name = method_name_node.text

        property_uncertainty_definitions = {}
        combined_uncertainty_definitions = {}

        cls.extract_uncertainty_definitions(node, namespace,
                                            property_uncertainty_definitions,
                                            combined_uncertainty_definitions)

        return_value.combined_uncertainty_definitions = combined_uncertainty_definitions
        return_value.property_uncertainty_definitions = property_uncertainty_definitions

        solvent_index_nodes = node.find('./ThermoML:Solvent//nOrgNum', namespace)

        if solvent_index_nodes is not None:
            for solvent_index_node in solvent_index_nodes:
                return_value.solvents.append(int(solvent_index_node))

        return return_value

    def set_value(self, value, uncertainty):
        """Set the value and uncertainty of this property, adding units if necessary.

        Parameters
        ----------
        value : float, unit.Quantity
            The value of the property
        uncertainty : float
            The uncertainty in the value.
        """
        value_quantity = value
        uncertainty_quantity = uncertainty

        if not isinstance(value_quantity, unit.Quantity):
            value_quantity = unit.Quantity(value, self.default_unit)
        if not isinstance(uncertainty_quantity, unit.Quantity):
            uncertainty_quantity = unit.Quantity(uncertainty, self.default_unit)

        self.value = value_quantity
        self.uncertainty = uncertainty_quantity


class ThermoMLPureOrMixtureData:
    """A wrapper around a ThermoML PureOrMixtureData node.
    """

    @staticmethod
    def extract_compound_indices(node, namespace, compounds):
        """Extract a list of ThermoMLCompounds from a PureOrMixtureData node.

        Parameters
        ----------
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.
        compounds :
            The extracted compounds.
        """

        component_nodes = node.findall('ThermoML:Component', namespace)
        compound_indices = []

        # Figure out which compounds are going to be associated with
        # the property entries.
        for component_node in component_nodes:

            index_node = component_node.find('./ThermoML:RegNum/*', namespace)

            compound_index = int(index_node.text)

            if compound_index not in compounds:

                logging.warning('A PureOrMixtureData entry depends on an '
                                'unsupported compound and has been ignored')

                return None

            if compound_index in compound_indices:

                raise RuntimeError('A ThermoML:PureOrMixtureData entry defines the same compound twice')

            compound_indices.append(compound_index)

        return compound_indices

    @staticmethod
    def extract_property_definitions(node, namespace):
        """Extract a list of ThermoMLProperty from a PureOrMixtureData node.

        Parameters
        ----------
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.

        Returns
        ----------
        dict(int, ThermoMLProperty)
            The extracted properties.
        """

        property_nodes = node.findall('ThermoML:Property', namespace)
        properties = {}

        for property_node in property_nodes:

            property_definition = ThermoMLProperty.from_xml_node(property_node, namespace)

            if property_definition is None:
                continue

            if property_definition.index in properties:

                raise RuntimeError('A ThermoML data set contains two '
                                   'properties with the same index')

            properties[property_definition.index] = property_definition

        return properties

    @staticmethod
    def extract_global_constraints(node, namespace, compounds):
        """Extract a list of ThermoMLConstraint from a PureOrMixtureData node.

        Parameters
        ----------
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.
        compounds : dict(int, ThermoMLCompound)
            A dictionary of the compounds this PureOrMixtureData was calculated for.

        Returns
        ----------
        dict(int, ThermoMLConstraint)
            The extracted constraints.
        """

        constraint_nodes = node.findall('ThermoML:Constraint', namespace)
        constraints = []

        for constraint_node in constraint_nodes:

            constraint = ThermoMLConstraint.from_node(constraint_node, namespace)

            if constraint is None or constraint.type is ThermoMLConstraintType.Undefined:
                logging.warning('An unsupported constraint has been ignored.')
                return None

            if constraint.compound_index is not None and \
               constraint.compound_index not in compounds:

                logging.warning('A constraint exists upon a non-existent compound and will be ignored.')
                return None

            if constraint.type.is_composition_constraint() and constraint.compound_index is None:

                logging.warning('An unsupported constraint has been ignored - composition constraints'
                                'need to have a corresponding compound_index.')

            constraints.append(constraint)

        return constraints

    @staticmethod
    def extract_variable_definitions(node, namespace, compounds):
        """Extract a list of ThermoMLVariableDefinition from a PureOrMixtureData node.

        Parameters
        ----------
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.
        compounds : dict(int, ThermoMLCompound)
            A dictionary of the compounds this PureOrMixtureData was calculated for.

        Returns
        ----------
        dict(int, ThermoMLVariableDefinition)
            The extracted constraints.
        """
        variable_nodes = node.findall('ThermoML:Variable', namespace)
        variables = {}

        for variable_node in variable_nodes:

            variable = ThermoMLVariableDefinition.from_node(variable_node, namespace)

            if variable is None or variable.type is ThermoMLConstraintType.Undefined:

                logging.warning('An unsupported variable has been ignored.')

                continue

            if variable.compound_index is not None and \
               variable.compound_index not in compounds:

                logging.warning('A constraint exists upon a non-existent compound and will be ignored.')

                continue

            if variable.type.is_composition_constraint() and variable.compound_index is None:

                logging.warning('An unsupported variable has been ignored - composition variables'
                                'need to have a corresponding compound_index.')

            variables[variable.index] = variable

        return variables

    @staticmethod
    def calculate_uncertainty(node, namespace, property_definition):

        # Look for a standard uncertainty..
        uncertainty_node = node.find('.//ThermoML:nCombStdUncertValue', namespace)

        if uncertainty_node is None:
            uncertainty_node = node.find('.//ThermoML:nStdUncertValue', namespace)

        # We have found a std. uncertainty
        if uncertainty_node is not None:
            return float(uncertainty_node.text)

        # Try to calculate uncertainty from a coverage factor if present
        if len(property_definition.combined_uncertainty_definitions) == 0 and \
           len(property_definition.property_uncertainty_definitions) == 0:

            return None

        combined = len(property_definition.combined_uncertainty_definitions) > 0

        prefix = ThermoMLCombinedUncertainty.prefix if combined \
            else ThermoMLPropertyUncertainty.prefix

        if combined:
            index_node = node.find('./ThermoML:CombinedUncertainty/ThermoML:nCombUncertAssessNum', namespace)
        else:
            index_node = node.find('./ThermoML:PropUncertainty/ThermoML:nUncertAssessNum', namespace)

        expanded_uncertainty_node = node.find('.//ThermoML:n' + prefix + 'ExpandUncertValue', namespace)

        if index_node is None or expanded_uncertainty_node is None:
            return None

        expanded_uncertainty = float(expanded_uncertainty_node.text)
        index = int(index_node.text)

        if combined and index not in property_definition.combined_uncertainty_definitions:
            return None

        if not combined and index not in property_definition.property_uncertainty_definitions:
            return None

        divisor = property_definition.combined_uncertainty_definitions[index].coverage_factor if combined \
            else property_definition.property_uncertainty_definitions[index].coverage_factor

        return expanded_uncertainty / divisor

    @staticmethod
    def _mole_fraction_constraints_to_mole_fractions(constraints, compounds):
        """Converts a set of `ThermoMLConstraint` to mole fractions.

        Parameters
        ----------
        constraints: list of ThermoMLConstraint
            The constraints to convert.
        compounds: dict of int and ThermoMLCompound
            The compounds in the system.

        Returns
        -------
        dict of int and float
            A dictionary of compound indices and mole fractions.
        """

        mol_fractions = {}

        number_of_constraints = 0
        total_mol_fraction = 0.0

        for constraint in constraints:

            if not constraint.type.is_composition_constraint():
                continue

            mole_fraction = constraint.value

            if isinstance(mole_fraction, unit.Quantity):
                mole_fraction = mole_fraction.value_in_unit(unit.dimensionless)

            mol_fractions[constraint.compound_index] = mole_fraction

            total_mol_fraction += mol_fractions[constraint.compound_index]
            number_of_constraints += 1

        if number_of_constraints == len(compounds) and not np.isclose(total_mol_fraction, 1.0):
            raise ValueError('The total mol fraction does not add to 1.0')

        elif number_of_constraints > len(compounds):
            raise ValueError('There are more concentration constraints than components.')

        elif number_of_constraints < len(compounds) - 1:
            raise ValueError('There are too many unknown mole fractions.')

        elif number_of_constraints == len(compounds) - 1:

            for compound_index in compounds:

                if compound_index in mol_fractions:
                    continue

                mol_fractions[compound_index] = 1.0 - total_mol_fraction

        else:
            raise ValueError('An unexpected edge case occurred when building the substance.')

        return mol_fractions

    @staticmethod
    def _mass_fraction_constraints_to_mole_fractions(constraints, compounds):
        """Converts a set of `ThermoMLConstraint` to mole fractions.

        Parameters
        ----------
        constraints: list of ThermoMLConstraint
            The constraints to convert.
        compounds: dict of int and ThermoMLCompound
            The compounds in the system.

        Returns
        -------
        dict of int and float
            A dictionary of compound indices and mole fractions.
        """

        from openforcefield.topology import Molecule

        mass_fractions = {}
        mole_fractions = {}

        number_of_constraints = 0
        total_mass_fraction = 0.0

        for constraint in constraints:

            if not constraint.type.is_composition_constraint():
                continue

            mass_fraction = constraint.value

            if isinstance(mass_fraction, unit.Quantity):
                mass_fraction = mass_fraction.value_in_unit(unit.dimensionless)

            mass_fractions[constraint.compound_index] = mass_fraction

            total_mass_fraction += mass_fraction
            number_of_constraints += 1

        if number_of_constraints == len(compounds) and not np.isclose(total_mass_fraction, 1.0):
            raise ValueError('The total mass fraction does not add to 1.0')

        elif number_of_constraints > len(compounds):
            raise ValueError('There are more concentration constraints than components.')

        elif number_of_constraints < len(compounds) - 1:
            raise ValueError('There are too many unknown mass fractions.')

        elif number_of_constraints == len(compounds) - 1:

            for compound_index in compounds:

                if compound_index not in mass_fractions:
                    continue

                mass_fractions[compound_index] = 1.0 - total_mass_fraction

        else:
            raise ValueError('An unexpected edge case occurred when building the substance.')

        base_mass = 1 * unit.gram

        for compound_index in compounds:

            compound_smiles = compounds[compound_index]
            compound_molecule = Molecule.from_smiles(compound_smiles)

            total_molecular_weight = sum([atom.mass for atom in compound_molecule.atoms])
            mole_fraction = base_mass * mass_fractions[compound_index] / total_molecular_weight

            mole_fractions[compound_index] = mole_fraction

        return mole_fractions

    @staticmethod
    def _molality_constraints_to_mole_fractions(constraints):
        raise NotImplementedError()

    @staticmethod
    def build_mixture(constraints, compounds):
        """Build a Substance object from the extracted constraints and compounds.

        Parameters
        ----------
        constraints : list of ThermoMLConstraint
            The ThermoML constraints.
        compounds : dict of int and ThermoMLCompound
            A dictionary of the compounds this PureOrMixtureData was calculated for.

        Returns
        ----------
        Substance
            The constructed mixture.
        """

        solvent_constraint_type = ThermoMLConstraintType.Undefined
        component_constraint_type = ThermoMLConstraintType.Undefined

        for constraint in constraints:

            if not constraint.type.is_composition_constraint():
                continue

            if (constraint.type == ThermoMLConstraintType.SolventMassFraction or
                constraint.type == ThermoMLConstraintType.SolventMoleFraction or
                constraint.type == ThermoMLConstraintType.SolventMolality):

                if solvent_constraint_type == ThermoMLConstraintType.Undefined:
                    solvent_constraint_type = constraint.type

                if solvent_constraint_type != constraint.type:

                    logging.warning(f'A property with different types of solvent composition constraints '
                                    f'was found - {solvent_constraint_type} vs {constraint.type}).')

                    return None

            else:

                if component_constraint_type == ThermoMLConstraintType.Undefined:
                    component_constraint_type = constraint.type

                if component_constraint_type != constraint.type:

                    logging.warning(f'A property with different types of composition constraints '
                                    f'was found - {component_constraint_type} vs {constraint.type}).')

                    return None

        if (component_constraint_type == ThermoMLConstraintType.Undefined and
            solvent_constraint_type == ThermoMLConstraintType.Undefined):

            component_constraint_type = ThermoMLConstraintType.ComponentMoleFraction

        elif (component_constraint_type == ThermoMLConstraintType.Undefined and
              solvent_constraint_type != ThermoMLConstraintType.Undefined):

            logging.warning(f'A property with only solvent composition '

                            f'constraints {solvent_constraint_type} was found.')

            return None

        if (component_constraint_type == ThermoMLConstraintType.ComponentMoleFraction and
            solvent_constraint_type == ThermoMLConstraintType.Undefined):

            mole_fractions = ThermoMLPureOrMixtureData._mole_fraction_constraints_to_mole_fractions(constraints,
                                                                                                    compounds)

        elif (component_constraint_type == ThermoMLConstraintType.ComponentMassFraction and
              solvent_constraint_type == ThermoMLConstraintType.Undefined):

            mole_fractions = ThermoMLPureOrMixtureData._mass_fraction_constraints_to_mole_fractions(constraints,
                                                                                                    compounds)

        else:

            logging.warning('An not implemented but supported composition constraint was found. '
                            'It will for now be assumed all mole fractions are equal to zero.')

            mole_fractions = {compound_index: 0.0 for compound_index in compounds}

        substance = Substance()

        for compound_index in compounds:

            compound = compounds[compound_index]

            if np.isclose(mole_fractions[compound_index], 0.0):
                continue

            substance.add_component(component=Substance.Component(smiles=compound.smiles),
                                    amount=Substance.MoleFraction(mole_fractions[compound_index]))

        return substance

    @staticmethod
    def extract_measured_properties(node, namespace,
                                    property_definitions,
                                    global_constraints,
                                    variable_definitions,
                                    compounds):

        """Extract the measured properties defined by a ThermoML PureOrMixtureData node.

        Parameters
        ----------
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.
        property_definitions
            The extracted property definitions.
        global_constraints
            The extracted constraints.
        variable_definitions
            The extracted variable definitions.
        compounds
            The extracted compounds.

        Returns
        ----------
        list(MeasuredPhysicalProperty)
            The extracted measured properties.
        """

        value_nodes = node.findall('ThermoML:NumValues', namespace)

        measured_properties = []

        # Each value_node corresponds to one MeasuredProperty
        for value_node in value_nodes:

            constraints = []

            temperature_constraint = None
            pressure_constraint = None

            for global_constraint in global_constraints:

                # constraint = copy.deepcopy(global_constraint)
                # constraint = json.loads(json.dumps(global_constraint))
                constraint = pickle.loads(pickle.dumps(global_constraint, -1))
                constraints.append(constraint)

                if constraint.type == ThermoMLConstraintType.Temperature:
                    temperature_constraint = constraint
                elif constraint.type == ThermoMLConstraintType.Pressure:
                    pressure_constraint = constraint

            # First extract the values of any variable constraints
            variable_nodes = value_node.findall('ThermoML:VariableValue', namespace)

            skip_entry = False

            for variable_node in variable_nodes:

                variable_index = int(variable_node.find('./ThermoML:nVarNumber', namespace).text)

                if variable_index not in variable_definitions:

                    # The property was constrained by an unsupported variable and
                    # so will be skipped for now.
                    skip_entry = True
                    break

                variable_definition = variable_definitions[variable_index]

                variable_value = float(variable_node.find('ThermoML:nVarValue', namespace).text)
                value_as_quantity = unit.Quantity(variable_value, variable_definition.default_unit)

                "Convert the 'variable' into a full constraint entry"
                constraint = ThermoMLConstraint.from_variable(variable_definition, value_as_quantity)
                constraints.append(constraint)

                if constraint.type == ThermoMLConstraintType.Temperature:
                    temperature_constraint = constraint
                elif constraint.type == ThermoMLConstraintType.Pressure:
                    pressure_constraint = constraint

            if skip_entry:
                continue

            # Extract the thermodynamic state that the property was measured at.
            if temperature_constraint is None:

                logging.warning('A property did not the temperature or the pressure it was measured '
                                'at and will be ignored.')
                continue

            temperature = temperature_constraint.value
            pressure = None if pressure_constraint is None else pressure_constraint.value

            thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)

            # Now extract the actual values of the measured properties, and their
            # uncertainties
            property_nodes = value_node.findall('ThermoML:PropertyValue', namespace)

            for property_node in property_nodes:

                property_index = int(property_node.find('./ThermoML:nPropNumber', namespace).text)

                if property_index not in property_definitions:

                    # Most likely the property was dropped earlier due to an unsupported phase / type
                    continue

                property_definition = property_definitions[property_index]

                uncertainty = ThermoMLPureOrMixtureData.calculate_uncertainty(property_node,
                                                                              namespace,
                                                                              property_definition)

                if uncertainty is None:

                    logging.warning('A property (' + str(property_definition.type) +
                                    ') without uncertainties was ignored')

                    continue

                # measured_property = copy.deepcopy(property_definition)
                # measured_property = json.loads(json.dumps(property_definition))
                measured_property = pickle.loads(pickle.dumps(property_definition, -1))
                measured_property.thermodynamic_state = thermodynamic_state

                property_value_node = property_node.find('.//ThermoML:nPropValue', namespace)

                measured_property.set_value(float(property_value_node.text),
                                            float(uncertainty))

                mixture = ThermoMLPureOrMixtureData.build_mixture(constraints, compounds)

                if mixture is None:

                    logging.warning('Could not build a mixture for a property (' +
                                    str(property_definition.type) + ').')

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
        node : Element
            The xml node to read.
        namespace : dict of str and str
            The xml namespace.
        compounds : dict of int and ThermoMLCompound
            A list of the already extracted `ThermoMLCompound`'s.

        Returns
        ----------
        list(MeasuredPhysicalProperty)
            A list of extracted properties.
        """

        # Figure out which compounds are going to be associated with
        # the property entries.
        compound_indices = ThermoMLPureOrMixtureData.extract_compound_indices(node, namespace, compounds)

        if compound_indices is None:
            # Most likely this entry depended on a non-parsable compound
            # and will be skipped entirely
            return None

        if len(compound_indices) == 0:

            logging.warning('A PureOrMixtureData entry with no compounds was ignored.')
            return None

        # Extract property definitions - values come later!
        property_definitions = ThermoMLPureOrMixtureData.extract_property_definitions(node, namespace)

        if len(property_definitions) == 0:

            logging.warning('A PureOrMixtureData entry with no properties was ignored. ' +
                            'Most likely this entry only contained unsupported properties.')

            return None

        phase_nodes = node.findall('./ThermoML:PhaseID/ThermoML:ePhase', namespace)

        all_phases = None

        for phase_node in phase_nodes:

            phase = phase_from_thermoml_string(phase_node.text)

            if phase == PropertyPhase.Undefined:
                # TODO: For now we just hope that the property defines the phase.
                #       This needs to be better supported however.
                logging.warning('A property was measured in an unsupported phase (' +
                                phase_node.text + ') and will be skipped.')

                return None

            all_phases = phase if all_phases is None else all_phases | phase

        for property_index in property_definitions:

            all_phases |= property_definitions[property_index].phase
            property_definitions[property_index].phase |= all_phases

        # Extract any constraints on the system e.g pressure, temperature
        global_constraints = ThermoMLPureOrMixtureData.extract_global_constraints(node, namespace, compounds)

        if global_constraints is None:
            return None

        # Extract any variables set on the system e.g pressure, temperature
        # Only the definition entry and not the value of the variable is extracted
        variable_definitions = ThermoMLPureOrMixtureData.extract_variable_definitions(node, namespace, compounds)

        if len(global_constraints) == 0 and len(variable_definitions) == 0:

            logging.warning('A PureOrMixtureData entry with no constraints was ignored.')
            return None

        used_compounds = {}

        for compound_index in compounds:

            if compound_index not in compound_indices:
                continue

            used_compounds[compound_index] = compounds[compound_index]

        measured_properties = ThermoMLPureOrMixtureData.extract_measured_properties(node, namespace,
                                                                                    property_definitions,
                                                                                    global_constraints,
                                                                                    variable_definitions,
                                                                                    used_compounds)

        return measured_properties


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

    def __init__(self):
        """Constructs a new ThermoMLDataSet object."""
        super().__init__()

    @classmethod
    def from_doi(cls, *doi_list):
        """Load a ThermoML data set from a list of DOIs

        Parameters
        ----------
        doi_list : *str
            The list of DOIs to pull data from

        Returns
        ----------
        ThermoMLDataSet
            The loaded data set.
        """
        return_value = None

        for doi in doi_list:

            # E.g https://trc.nist.gov/ThermoML/10.1016/j.jct.2016.12.009.xml
            doi_url = 'https://trc.nist.gov/ThermoML/' + doi + '.xml'

            data_set = cls._from_url(doi_url, MeasurementSource(doi=doi))

            if data_set is None or len(data_set.properties) == 0:
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
        url_list : *str
            The list of URLs to pull data from

        Returns
        ----------
        ThermoMLDataSet
            The loaded data set.
        """

        return_value = None

        for url in url_list:

            data_set = cls._from_url(url)

            if data_set is None or len(data_set.properties) == 0:
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
        url : str
            The URL to pull data from
        source : Source, optional
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

            with urlopen(url) as response:
                return_value = cls.from_xml(response.read(), source)

        except HTTPError:
            logging.warning('WARNING: No ThermoML file could not be found at ' + url)

        return return_value

    @classmethod
    def from_file(cls, *file_list):
        """Load a ThermoML data set from a list of files

        Parameters
        ----------
        file_list : *str
            The list of files to pull data from

        Returns
        ----------
        ThermoMLDataSet
            The loaded data set.
        """
        return_value = None
        counter = 0

        for file in file_list:

            data_set = cls._from_file(file)

            logging.info('Reading file ' + str(counter + 1) + ' of ' + str(len(file_list)) + ' (' + file + ')')
            counter += 1

            if data_set is None or len(data_set.properties) == 0:
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
        path : str
            The file path to pull data from

        Returns
        ----------
        ThermoMLDataSet
            The loaded data set.
        """
        source = MeasurementSource(reference=path)
        return_value = None

        try:

            with open(path) as file:
                return_value = ThermoMLDataSet.from_xml(file.read(), source)

        except FileNotFoundError:

            logging.warning('No ThermoML file could not be found at ' + path)

        return return_value

    @classmethod
    def from_xml(cls, xml, source):
        """Load a ThermoML data set from an xml object.

        Parameters
        ----------
        xml : str
            The xml string to parse.
        source : Source
            The source of the xml object.

        Returns
        ----------
        ThermoMLDataSet
            The loaded ThermoML data set.
        """
        root_node = ElementTree.fromstring(xml)

        if root_node is None:
            logging.warning('The ThermoML XML document could not be parsed.')
            return None

        if root_node.tag.find('DataReport') < 0:
            logging.warning('The ThermoML XML document does not contain the expected root node.')
            return None

        # Extract the namespace that will prefix all type names
        namespace_string = re.search(r'{.*\}', root_node.tag).group(0)[1:-1]
        namespace = {'ThermoML': namespace_string}

        return_value = ThermoMLDataSet()
        compounds = {}

        # Extract the base compounds present in the xml file
        for node in root_node.findall('ThermoML:Compound', namespace):

            compound = ThermoMLCompound.from_xml_node(node, namespace)

            if compound is None:
                continue

            if compound.index in compounds:
                raise RuntimeError('A ThermoML data set contains two '
                                   'compounds with the same index')

            compounds[compound.index] = compound

        # Pull out any and all properties in the file.
        for node in root_node.findall('ThermoML:PureOrMixtureData', namespace):

            properties = ThermoMLPureOrMixtureData.from_xml_node(node,
                                                                 namespace,
                                                                 compounds)

            if properties is None or len(properties) == 0:
                continue

            for measured_property in properties:

                substance_id = measured_property.substance.identifier

                if substance_id not in return_value._properties:
                    return_value._properties[substance_id] = []

                if measured_property.type is None:
                    raise ValueError('An unexepected property type managed to slip through the cracks.')

                final_property = measured_property.type()

                final_property.value = measured_property.value
                final_property.uncertainty = measured_property.uncertainty

                final_property.phase = measured_property.phase

                final_property.thermodynamic_state = measured_property.thermodynamic_state
                final_property.substance = measured_property.substance

                final_property.source = source

                return_value._properties[substance_id].append(final_property)

        return_value._sources.append(source)

        return return_value
