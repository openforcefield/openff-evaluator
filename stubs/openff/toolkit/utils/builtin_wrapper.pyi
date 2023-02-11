from _typeshed import Incomplete
from openff.toolkit.utils import base_wrapper

class BuiltInToolkitWrapper(base_wrapper.ToolkitWrapper):
    def __init__(self) -> None: ...
    def assign_partial_charges(
        self,
        molecule,
        partial_charge_method: Incomplete | None = ...,
        use_conformers: Incomplete | None = ...,
        strict_n_conformers: bool = ...,
        normalize_partial_charges: bool = ...,
        _cls: Incomplete | None = ...,
    ) -> None: ...
