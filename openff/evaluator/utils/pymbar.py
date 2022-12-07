"""
Helper functions to simultaneously support pymbar 3 and 4.
"""
try:
    from pymbar import pymbar  # noqa

    _PYMBAR_MAJOR_VERSION = 3
except ImportError:
    import pymbar  # noqa

    _PYMBAR_MAJOR_VERSION = 4


def __getattr__(name):
    if name == "detect_equilibration":
        if _PYMBAR_MAJOR_VERSION == 3:
            from pymbar.timeseries import detectEquilibration as detect_equilibration
        elif _PYMBAR_MAJOR_VERSION == 4:
            from pymbar.timeseries import detect_equilibration
        return detect_equilibration
    raise AttributeError
