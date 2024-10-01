"""
A collection of density physical property definitions.
"""

from openff.units import unit

from openff.evaluator.datasets import PhysicalProperty


class HostGuestBindingAffinity(PhysicalProperty):
    """A class representation of a host-guest binding affinity property"""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole
