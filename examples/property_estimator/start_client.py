#!/usr/bin/env python
import logging

from propertyestimator import client
from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.datasets import ThermoMLDataSet
from propertyestimator.workflow import WorkflowOptions
from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def compute_estimate_sync():
    """Submit calculations to a running server instance"""
    from openforcefield.typing.engines import smirnoff

    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))
    # Load in the force field to use.
    force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml')

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server, and wait for the results.
    result = property_estimator.request_estimate(data_set, force_field)

    logging.info('The server has returned a response: {}'.format(result))


def compute_estimate_async():
    """Submit calculations to a running server instance"""
    from openforcefield.typing.engines import smirnoff

    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_dielectric.xml'))
    # Load in the force field to use.
    force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml')

    # new_property_0 = copy.deepcopy(data_set.properties['COCCO{1.0}'][0])
    # new_property_0.thermodynamic_state.temperature -= 2.0 * unit.kelvin
    # new_property_0.id = str(uuid4())
    #
    # data_set.properties['COCCO{1.0}'].append(new_property_0)
    #
    # new_property_1 = copy.deepcopy(data_set.properties['COCCO{1.0}'][0])
    # new_property_1.thermodynamic_state.temperature += 2.0 * unit.kelvin
    # new_property_1.id = str(uuid4())
    #
    # data_set.properties['COCCO{1.0}'].append(new_property_1)

    # Modify the submission options
    submission_options = PropertyEstimatorOptions()

    workflow_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.RelativeUncertainty,
                                       relative_uncertainty_fraction=100000)

    submission_options.workflow_options = {
        'Density': workflow_options,
        'Dielectric': workflow_options,
        'EnthalpyOfMixing': workflow_options
    }
    # submission_options.allowed_calculation_layers = ['SimulationLayer']
    submission_options.allowed_calculation_layers = ['ReweightingLayer']

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server.
    request = property_estimator.request_estimate(data_set, force_field, submission_options)

    logging.info('Request info: {}'.format(str(request)))

    # Wait for the results.
    result = request.results(synchronous=True)

    logging.info('The server has returned a response: {}'.format(result.json()))


if __name__ == "__main__":
    compute_estimate_async()
