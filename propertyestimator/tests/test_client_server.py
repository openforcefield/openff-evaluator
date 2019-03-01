"""
Units tests for propertyestimator.client and server
"""
import tempfile
from os import path

from propertyestimator.backends import DaskLocalClusterBackend, ComputeResources
from propertyestimator.client import PropertyEstimatorClient, PropertyEstimatorOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import Density
from propertyestimator.server import PropertyEstimatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.exceptions import PropertyEstimatorException


def test_estimate_request():
    """Test sending an estimator request to a server."""

    from openforcefield.typing.engines import smirnoff

    with tempfile.TemporaryDirectory() as temporary_directory:

        storage_directory = path.join(temporary_directory, 'storage')
        working_directory = path.join(temporary_directory, 'working')

        dummy_property = create_dummy_property(Density)

        dummy_data_set = PhysicalPropertyDataSet()
        dummy_data_set.properties[dummy_property.substance.identifier] = [dummy_property]

        force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

        calculation_backend = DaskLocalClusterBackend(1, ComputeResources())
        storage_backend = LocalFileStorage(storage_directory)

        PropertyEstimatorServer(calculation_backend, storage_backend, working_directory=working_directory)

        property_estimator = PropertyEstimatorClient()
        options = PropertyEstimatorOptions(allowed_calculation_layers=[])

        request = property_estimator.request_estimate(dummy_data_set, force_field, options)
        result = request.results(synchronous=True, polling_interval=0)

        assert result is not PropertyEstimatorException
