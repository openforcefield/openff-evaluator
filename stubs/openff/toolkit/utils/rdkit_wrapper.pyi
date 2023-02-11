from typing import List, Tuple

from _typeshed import Incomplete
from openff.toolkit.topology.molecule import Atom, Bond, Molecule
from openff.toolkit.utils import base_wrapper
from openff.units import Quantity, unit

class RDKitToolkitWrapper(base_wrapper.ToolkitWrapper):
    def __init__(self) -> None: ...
    @property
    def toolkit_file_write_formats(self): ...
    @classmethod
    def is_available(cls): ...
    def from_object(
        self, obj, allow_undefined_stereo: bool = ..., _cls: Incomplete | None = ...
    ): ...
    def from_pdb_and_smiles(
        self,
        file_path,
        smiles,
        allow_undefined_stereo: bool = ...,
        _cls: Incomplete | None = ...,
    ): ...
    def from_file(
        self,
        file_path,
        file_format,
        allow_undefined_stereo: bool = ...,
        _cls: Incomplete | None = ...,
    ): ...
    def from_file_obj(
        self,
        file_obj,
        file_format,
        allow_undefined_stereo: bool = ...,
        _cls: Incomplete | None = ...,
    ): ...
    def to_file_obj(self, molecule, file_obj, file_format) -> None: ...
    def to_file(self, molecule, file_path, file_format) -> None: ...
    def enumerate_stereoisomers(
        self,
        molecule,
        undefined_only: bool = ...,
        max_isomers: int = ...,
        rationalise: bool = ...,
    ): ...
    def enumerate_tautomers(self, molecule, max_states: int = ...): ...
    def canonical_order_atoms(self, molecule): ...
    def to_smiles(
        self,
        molecule,
        isomeric: bool = ...,
        explicit_hydrogens: bool = ...,
        mapped: bool = ...,
    ): ...
    def from_smiles(
        self,
        smiles,
        hydrogens_are_explicit: bool = ...,
        allow_undefined_stereo: bool = ...,
        _cls: Incomplete | None = ...,
    ): ...
    def from_inchi(
        self, inchi, allow_undefined_stereo: bool = ..., _cls: Incomplete | None = ...
    ): ...
    def generate_conformers(
        self,
        molecule,
        n_conformers: int = ...,
        rms_cutoff: Incomplete | None = ...,
        clear_existing: bool = ...,
        _cls: Incomplete | None = ...,
        make_carboxylic_acids_cis: bool = ...,
    ) -> None: ...
    def assign_partial_charges(
        self,
        molecule,
        partial_charge_method: Incomplete | None = ...,
        use_conformers: Incomplete | None = ...,
        strict_n_conformers: bool = ...,
        normalize_partial_charges: bool = ...,
        _cls: Incomplete | None = ...,
    ) -> None: ...
    def apply_elf_conformer_selection(
        self,
        molecule: Molecule,
        percentage: float = ...,
        limit: int = ...,
        rms_tolerance: Quantity = ...,
    ): ...
    def from_rdkit(
        self,
        rdmol,
        allow_undefined_stereo: bool = ...,
        hydrogens_are_explicit: bool = ...,
        _cls: Incomplete | None = ...,
    ): ...
    to_rdkit_cache: Incomplete
    def to_rdkit(self, molecule, aromaticity_model=...): ...
    def to_inchi(self, molecule, fixed_hydrogens: bool = ...): ...
    def to_inchikey(self, molecule, fixed_hydrogens: bool = ...): ...
    def get_tagged_smarts_connectivity(self, smarts): ...
    def find_smarts_matches(
        self,
        molecule: Molecule,
        smarts: str,
        aromaticity_model: str = ...,
        unique: bool = ...,
    ) -> List[Tuple[int, ...]]: ...
    def atom_is_in_ring(self, atom: Atom) -> bool: ...
    def bond_is_in_ring(self, bond: Bond) -> bool: ...
