def test_openeye_is_available():
    import importlib
    oechem = importlib.import_module('openeye', 'oechem')
    
    if not oechem.OEChemIsLicensed():
        raise ImportError
