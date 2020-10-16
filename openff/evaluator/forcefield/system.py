from typing import TYPE_CHECKING

from openff.evaluator.forcefield import ForceFieldSource
from openff.evaluator.substances import Substance
from openff.evaluator.utils.serialization import TypedBaseModel

if TYPE_CHECKING:

    from openforcefield.topology import Topology
    from simtk.openmm import System


class ParameterizedSystem(TypedBaseModel):
    """An object model which stores information about a parameterized system,
    including the composition of the system, the original force field source, a
    path to the topology and a path to the parameterized system object."""

    @property
    def substance(self) -> Substance:
        return self._substance

    @property
    def force_field(self) -> ForceFieldSource:
        return self._force_field

    @property
    def topology_path(self) -> str:
        return self._topology_path

    @property
    def topology(self) -> "Topology":

        from openforcefield.topology import Molecule, Topology
        from simtk.openmm import app

        pdb_file = app.PDBFile(self._topology_path)

        topology = Topology.from_openmm(
            pdb_file.topology,
            unique_molecules=[
                Molecule.from_smiles(smiles=component.smiles)
                for component in self._substance.components
            ],
        )

        return topology

    @property
    def system_path(self) -> str:
        return self._system_path

    @property
    def system(self) -> "System":
        from simtk import openmm

        with open(self._system_path) as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        return system

    def __init__(
        self,
        substance: Substance = None,
        force_field: ForceFieldSource = None,
        topology_path: str = None,
        system_path: str = None,
    ):

        self._substance = substance
        self._force_field = force_field

        self._topology_path = topology_path
        self._system_path = system_path

    def __getstate__(self):

        return {
            "substance": self._substance,
            "force_field": self._force_field,
            "topology_path": self._topology_path,
            "system_path": self._system_path,
        }

    def __setstate__(self, state):

        self._substance = state["substance"]
        self._force_field = state["force_field"]
        self._topology_path = state["topology_path"]
        self._system_path = state["system_path"]
