"""
A collection of protocols for assigning force field parameters to molecular systems.
"""

import logging
from os import path

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

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self, value):
        """The file path to the coordinate file which defines the system to which the
        force field parameters will be assigned."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system."""
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

        # outputs
        self._system_path = None

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

        for component in self._substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)

            if molecule is None:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            unique_molecules.append(molecule)

        topology = Topology.from_openmm(pdb_file.topology, unique_molecules=unique_molecules)
        system = force_field.create_openmm_system(topology)

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
