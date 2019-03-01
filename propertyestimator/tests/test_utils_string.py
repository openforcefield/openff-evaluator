"""
Units tests for propertyestimator.utils.statistics
"""
import pytest
from simtk import unit

from propertyestimator.utils import string


def test_valid_extract_variable_index_and_name():

    valid_path = 'variable_name[index_string]'

    variable_name, index = string.extract_variable_index_and_name(valid_path)
    assert variable_name == 'variable_name' and index == 'index_string'

    valid_path = 'variable_name[1000]'

    variable_name, index = string.extract_variable_index_and_name(valid_path)
    assert variable_name == 'variable_name' and index == '1000'


@pytest.mark.parametrize("invalid_path", [
    'val[0]ue',
    '[0]value',
    'value]0[',
    'value[0',
    'value0]',
    'value[0][1]',
    'value[]'
])
def test_invalid_extract_variable_index_and_name(invalid_path):

    with pytest.raises(ValueError):
        string.extract_variable_index_and_name(invalid_path)
