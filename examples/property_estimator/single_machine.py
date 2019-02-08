#!/usr/bin/env python

import logging
import shutil
from os import path

from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
from propertyestimator import client, server
from propertyestimator.backends import PropertyEstimatorBackendResources, DaskLocalClusterBackend
from propertyestimator.storage import LocalFileStorage
from openforcefield.typing.engines import smirnoff
from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def run_property_estimator():
    """An integrated test of the property estimator"""

    setup_timestamp_logging()

    working_directory = 'working_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    complete_data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/JCT/j.jct.2008.12.004.xml'))

    data_set = PhysicalPropertyDataSet()

    data_set.properties['[CH3].C1CCCCC1{0.074}|[H].CCCCCC{0.926}'] = \
        complete_data_set.properties['[CH3].C1CCCCC1{0.074}|[H].CCCCCC{0.926}']

    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/single_density.xml'))
    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/single_dielectric.xml'))

    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/single_density.xml'),
    #                                           get_data_filename('properties/single_dielectric.xml'))

    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/density_dielectric.xml'))
    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/two_species.xml'))
    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/binary.xml'))

    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    calculation_backend = DaskLocalClusterBackend(1, 1, PropertyEstimatorBackendResources(2, 0))
    storage_backend = LocalFileStorage()

    property_server = server.PropertyCalculationRunner(calculation_backend,
                                                       storage_backend,
                                                       working_directory=working_directory)

    property_estimator = client.PropertyEstimator()
    property_estimator.submit_computations(data_set, force_field)

    property_server.run_until_complete()

    for property_id in property_server.finished_calculations:

        logging.info('Calculation {}:'.format(property_id))
        logging.info('{}'.format(property_server.finished_calculations[property_id].json()))


if __name__ == "__main__":
    run_property_estimator()
