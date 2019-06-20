"""
A collection of protocols for assigning force field parameters to molecular systems.
"""

import logging
from enum import Enum
from os import path

import numpy as np
from simtk import unit
from simtk.openmm import app

from propertyestimator.substances import Substance
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class BuildSmirnoffSystem(BaseProtocol):
    """Parametrise a set of molecules with a given smirnoff force field.
    """
    class WaterModel(Enum):
        """An enum which describes which water model is being
        used, so that correct charges can be applied.

        Warnings
        --------
        This is only a temporary addition until library charges
        are introduced into the openforcefield toolkit.
        """
        TIP3P = 'TIP3P'

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self, value):
        """The file path to the coordinate file which defines the system to which the
        force field parameters will be assigned."""
        pass

    @protocol_input(list)
    def charged_molecule_paths(self):
        """File paths to mol2 files which contain the charges assigned to molecules
        in the system. This input is helpful when dealing with large molecules (such
        as hosts in host-guest binding calculations) whose charges may by needed
        in multiple places, and hence should only be calculated once."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system."""
        pass

    @protocol_input(WaterModel)
    def water_model(self):
        """The water model to apply, if any water molecules
        are present.

        Warnings
        --------
        This is only a temporary addition until library charges
        are introduced into the openforcefield toolkit.
        """
        pass

    @protocol_input(bool)
    def apply_known_charges(self):
        """If true, formal the formal charges of ions, and
        the charges of the selected water model will be
        automatically applied to any matching molecules in the
        system."""
        pass

    @protocol_input(bool)
    def allow_missing_parameters(self):
        """If true, no exception will be raised if no parameters
        have been assigned to any of the molecular interactions
        (e.g. if a particular bond has not been assigned a parameter
        because that chemical environment is not covered by the
        force field)."""
        pass

    @protocol_output(str)
    def system_path(self):
        """The assigned system."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # inputs
        self._force_field_path = None
        self._coordinate_file_path = None
        self._substance = None

        self._water_model = BuildSmirnoffSystem.WaterModel.TIP3P
        self._apply_known_charges = True

        self._allow_missing_parameters = False

        self._charged_molecule_paths = []

        # outputs
        self._system_path = None

    @staticmethod
    def _generate_known_charged_molecules():
        """Generates a set of molecules whose charges are known a priori,
        such as ions, for use in parameterised systems.

        Notes
        -----
        These are solely to be used as a work around until library charges
        are fully implemented in the openforcefield toolkit.

        Todos
        -----
        Remove this method when library charges are fully implemented in
        the openforcefield toolkit.

        Returns
        -------
        list of openforcefield.topology.Molecule
            The molecules with assigned charges.
        """
        from openforcefield.topology import Molecule

        sodium = Molecule.from_smiles('[Na+]')
        sodium.partial_charges = np.array([1.0]) * unit.elementary_charge

        potassium = Molecule.from_smiles('[K+]')
        potassium.partial_charges = np.array([1.0]) * unit.elementary_charge

        calcium = Molecule.from_smiles('[Ca+2]')
        calcium.partial_charges = np.array([2.0]) * unit.elementary_charge

        chlorine = Molecule.from_smiles('[Cl-]')
        chlorine.partial_charges = np.array([-1.0]) * unit.elementary_charge

        water = Molecule.from_smiles('O')
        water.partial_charges = np.array([-0.834, 0.417, 0.417]) * unit.elementary_charge

        return [sodium, potassium, calcium, chlorine, water]

    def execute(self, directory, available_resources):

        from openforcefield.typing.engines.smirnoff import ForceField
        from openforcefield.topology import Molecule, Topology

        logging.info('Generating topology: ' + self.id)

        pdb_file = app.PDBFile(self._coordinate_file_path)

        force_field = None

        try:

            force_field = ForceField(self._force_field_path, allow_cosmetic_attributes=True)

        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='{} could not load the ForceField: {}'.format(self.id, e))

        unique_molecules = []
        charged_molecules = []

        if self._apply_known_charges:
            charged_molecules = self._generate_known_charged_molecules()

        # Load in any additional, user specified charged molecules.
        for charged_molecule_path in self._charged_molecule_paths:

            charged_molecule = Molecule.from_file(charged_molecule_path, 'MOL2')
            charged_molecules.append(charged_molecule)

        for component in self._substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)

            if molecule is None:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            unique_molecules.append(molecule)

        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)

        if len(charged_molecules) > 0:
            system = force_field.create_openmm_system(topology,
                                                      allow_missing_parameters=self._allow_missing_parameters,
                                                      charge_from_molecules=charged_molecules)
        else:
            system = force_field.create_openmm_system(topology,
                                                      allow_missing_parameters=self._allow_missing_parameters)

        if system is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Failed to create a system from the'
                                                       'provided topology and molecules')

        from simtk.openmm import XmlSerializer
        system_xml = XmlSerializer.serialize(system)

        self._system_path = path.join(directory, 'system.xml')

        with open(self._system_path, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

        logging.info('Topology generated: ' + self.id)

        return self._get_output_dictionary()
