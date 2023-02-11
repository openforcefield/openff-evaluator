from _typeshed import Incomplete
from openff.toolkit.utils import base_wrapper

class AmberToolsToolkitWrapper(base_wrapper.ToolkitWrapper):
    def __init__(self) -> None: ...
    @staticmethod
    def is_available(): ...
    def assign_partial_charges(
        self,
        molecule,
        partial_charge_method: Incomplete | None = ...,
        use_conformers: Incomplete | None = ...,
        strict_n_conformers: bool = ...,
        normalize_partial_charges: bool = ...,
        _cls: Incomplete | None = ...,
    ) -> None: ...
    def assign_fractional_bond_orders(
        self,
        molecule,
        bond_order_model: Incomplete | None = ...,
        use_conformers: Incomplete | None = ...,
        _cls: Incomplete | None = ...,
    ) -> None: ...
