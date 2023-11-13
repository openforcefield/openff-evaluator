class Dummy1:
    pass

class Dummy2:
    raise ImportError(
        "This plugin should not be loaded; raise an ImportError to mock this."
    )