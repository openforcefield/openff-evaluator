import pytest

from openff.evaluator.datasets.taproom import TaproomDataSet
from openff.evaluator.datasets.taproom.taproom import TaproomSource
from openff.evaluator.utils.exceptions import MissingOptionalDependency

try:
    import openeye.oechem
except ImportError:
    openeye = None


@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_taproom():

    data_set = TaproomDataSet(host_codes=["acd"], guest_codes=["bam"])

    assert len(data_set) == 1
    assert isinstance(data_set.properties[0].source, TaproomSource)

    assert data_set.properties[0].source.host_identifier == "acd"
    assert data_set.properties[0].source.guest_identifier == "bam"


@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_taproom_missing_oe_license(monkeypatch):

    from openeye import oechem

    def mock_return():
        return False

    monkeypatch.setattr(oechem, "OEChemIsLicensed", mock_return)

    with pytest.raises(MissingOptionalDependency) as error_info:
        TaproomDataSet()

    assert error_info.value.library_name == "openeye.oechem"
    assert error_info.value.license_issue


@pytest.mark.skipif(
    openeye is not None or openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye must not be present for this test.",
)
def test_taproom_missing_oe():

    with pytest.raises(MissingOptionalDependency) as error_info:
        TaproomDataSet()

    assert error_info.value.library_name == "openeye.oechem"
    assert not error_info.value.license_issue
