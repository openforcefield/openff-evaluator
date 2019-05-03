"""
A collection of protocols for assigning force field parameters to molecular systems.
"""

import logging
import pickle
from os import path

from simtk import unit
from simtk.openmm import app

from propertyestimator.substances import Substance
from propertyestimator.utils import create_molecule_from_smiles
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import deserialize_force_field
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

    @protocol_input(unit.Quantity)
    def nonbonded_cutoff(self):
        """The cutoff after which non-bonded interactions are truncated."""
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

        self._nonbonded_cutoff = 1.0 * unit.nanometer

        # outputs
        self._system_path = None

    def execute(self, directory, available_resources):

        logging.info('Generating topology: ' + self.id)

        pdb_file = app.PDBFile(self._coordinate_file_path)

        force_field = None

        try:

            with open(self._force_field_path, 'rb') as file:
                force_field = deserialize_force_field(pickle.load(file))

        except pickle.UnpicklingError:

            try:

                from openforcefield.typing.engines.smirnoff import ForceField
                force_field = ForceField(self._force_field_path)

            except Exception as e:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not load the ForceField: {}'.format(self.id, e))

        molecules = []

        for component in self._substance.components:

            molecule = create_molecule_from_smiles(component.smiles, 0)

            if molecule is None:
                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            molecules.append(molecule)

        from openforcefield.typing.engines import smirnoff

        system = force_field.createSystem(pdb_file.topology,
                                          molecules,
                                          nonbondedMethod=smirnoff.PME,
                                          nonbondedCutoff=self._nonbonded_cutoff,
                                          chargeMethod='OECharges_AM1BCCSym')

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
