"""A temporary fix for ensuring processes run correctly
within pytest.
"""


def pytest_configure(config):
    import propertyestimator

    propertyestimator._called_from_test = True


def pytest_unconfigure(config):
    import propertyestimator  # This was missing from the manual

    del propertyestimator._called_from_test
