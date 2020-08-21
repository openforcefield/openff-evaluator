from openff.evaluator.datasets.taproom import TaproomDataSet
from openff.evaluator.datasets.taproom.taproom import TaproomSource


def test_taproom():

    data_set = TaproomDataSet(host_codes=["acd"], guest_codes=["bam"])

    assert len(data_set) == 1
    assert isinstance(data_set.properties[0].source, TaproomSource)

    assert data_set.properties[0].source.host_identifier == "acd"
    assert data_set.properties[0].source.guest_identifier == "bam"
