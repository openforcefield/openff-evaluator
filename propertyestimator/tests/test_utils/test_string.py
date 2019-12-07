"""
Units tests for propertyestimator.utils.statistics
"""
import tempfile
from os import path

import pytest

from propertyestimator.utils import string
from propertyestimator.utils.string import sanitize_smiles_file_name


def test_valid_extract_variable_index_and_name():

    valid_path = "variable_name[index_string]"

    variable_name, index = string.extract_variable_index_and_name(valid_path)
    assert variable_name == "variable_name" and index == "index_string"

    valid_path = "variable_name[1000]"

    variable_name, index = string.extract_variable_index_and_name(valid_path)
    assert variable_name == "variable_name" and index == "1000"


@pytest.mark.parametrize(
    "invalid_path",
    [
        "val[0]ue",
        "[0]value",
        "value]0[",
        "value[0",
        "value0]",
        "value[0][1]",
        "value[]",
    ],
)
def test_invalid_extract_variable_index_and_name(invalid_path):

    with pytest.raises(ValueError):
        string.extract_variable_index_and_name(invalid_path)


@pytest.mark.parametrize(
    "smiles_pattern", [r"C/C=C/C=C/COC(=O)", r"CCOC(=O)/C=C(/C)\O"]
)
def test_sanitize_smiles_file_name(smiles_pattern):

    smiles_file_name = f"file_{smiles_pattern}.json"
    sanitized_smiles_file_name = sanitize_smiles_file_name(smiles_file_name)

    assert sanitized_smiles_file_name.find("/") < 0

    with tempfile.TemporaryDirectory() as temporary_directory:

        with open(
            path.join(temporary_directory, sanitized_smiles_file_name), "w"
        ) as file:
            file.write(smiles_pattern)
