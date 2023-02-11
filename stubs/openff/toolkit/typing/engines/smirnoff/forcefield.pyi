from typing import List, Optional, Tuple, Union

import openmm
from _typeshed import Incomplete
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.utils.base_wrapper import ToolkitWrapper
from openff.toolkit.utils.exceptions import (
    ParameterHandlerRegistrationError as ParameterHandlerRegistrationError,
)
from openff.toolkit.utils.exceptions import (
    PartialChargeVirtualSitesError as PartialChargeVirtualSitesError,
)
from openff.toolkit.utils.exceptions import (
    SMIRNOFFAromaticityError as SMIRNOFFAromaticityError,
)
from openff.toolkit.utils.exceptions import SMIRNOFFParseError as SMIRNOFFParseError
from openff.toolkit.utils.exceptions import SMIRNOFFVersionError as SMIRNOFFVersionError
from openff.toolkit.utils.toolkit_registry import ToolkitRegistry
from openff.units import Quantity, unit

def get_available_force_fields(full_paths: bool = ...): ...

MAX_SUPPORTED_VERSION: str

class ForceField:
    disable_version_check: Incomplete
    def __init__(
        self,
        *sources,
        aromaticity_model=...,
        parameter_handler_classes: Incomplete | None = ...,
        parameter_io_handler_classes: Incomplete | None = ...,
        disable_version_check: bool = ...,
        allow_cosmetic_attributes: bool = ...,
        load_plugins: bool = ...
    ) -> None: ...
    @property
    def aromaticity_model(self): ...
    @aromaticity_model.setter
    def aromaticity_model(self, aromaticity_model) -> None: ...
    @property
    def author(self): ...
    @author.setter
    def author(self, author) -> None: ...
    @property
    def date(self): ...
    @date.setter
    def date(self, date) -> None: ...
    def register_parameter_handler(self, parameter_handler) -> None: ...
    def register_parameter_io_handler(self, parameter_io_handler) -> None: ...
    @property
    def registered_parameter_handlers(self) -> List[str]: ...
    def get_parameter_handler(
        self,
        tagname,
        handler_kwargs: Incomplete | None = ...,
        allow_cosmetic_attributes: bool = ...,
    ): ...
    def get_parameter_io_handler(self, io_format): ...
    def deregister_parameter_handler(self, handler) -> None: ...
    def parse_sources(self, sources, allow_cosmetic_attributes: bool = ...) -> None: ...
    def parse_smirnoff_from_source(self, source) -> dict: ...
    def to_string(
        self, io_format: str = ..., discard_cosmetic_attributes: bool = ...
    ): ...
    def to_file(
        self,
        filename,
        io_format: Incomplete | None = ...,
        discard_cosmetic_attributes: bool = ...,
    ) -> None: ...
    def create_openmm_system(
        self,
        topology: Topology,
        *,
        return_topology: bool = ...,
        toolkit_registry: Optional[Union["ToolkitRegistry", "ToolkitWrapper"]] = ...,
        charge_from_molecules: Optional[List["Molecule"]] = ...,
        partial_bond_orders_from_molecules: Optional[List["Molecule"]] = ...,
        allow_nonintegral_charges: bool = ...
    ) -> Union["openmm.System", Tuple["openmm.System", "Topology"]]: ...
    def create_interchange(
        self,
        topology: Topology,
        toolkit_registry: Optional[Union["ToolkitRegistry", "ToolkitWrapper"]] = ...,
        charge_from_molecules: Optional[List["Molecule"]] = ...,
        partial_bond_orders_from_molecules: Optional[List["Molecule"]] = ...,
        allow_nonintegral_charges: bool = ...,
    ): ...
    def label_molecules(self, topology): ...
    def get_partial_charges(self, molecule: Molecule, **kwargs) -> Quantity: ...
    def __getitem__(self, val): ...
    def __hash__(self): ...
