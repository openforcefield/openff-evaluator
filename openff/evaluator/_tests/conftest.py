"""A temporary fix for ensuring processes run correctly
within pytest.
"""


def pytest_configure(config):
    from openff import evaluator

    evaluator._called_from_test = True


def pytest_unconfigure(config):
    from openff import evaluator

    del evaluator._called_from_test
