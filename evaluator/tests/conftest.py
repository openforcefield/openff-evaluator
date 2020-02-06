"""A temporary fix for ensuring processes run correctly
within pytest.
"""


def pytest_configure(config):
    import evaluator

    evaluator._called_from_test = True


def pytest_unconfigure(config):
    import evaluator  # This was missing from the manual

    del evaluator._called_from_test
