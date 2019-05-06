"""
A collection of binding affinity physical property definitions.
"""

from propertyestimator.properties import PhysicalProperty
from propertyestimator.properties.plugins import register_estimable_property


@register_estimable_property()
class HostGuestBindingAffinity(PhysicalProperty):
    """A class representation of a dielectric property"""

    @property
    def multi_component_property(self):
        return False

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return HostGuestBindingAffinity.get_default_simulation_workflow_schema(options)

        return None

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        return HostGuestBindingAffinity._get_yank_simulation_workflow()

    @staticmethod
    def _get_yank_simulation_workflow():

        return None
