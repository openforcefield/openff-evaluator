"""
A collection of utilities for manipulating strings.
"""


def extract_variable_index_and_name(string):
    """Takes a string of the form variable_name[index] and
    returns the variable_name, and index.

    The method will return a `ValueError` if the string does not
    contain a valid set of access brackets, or if the string contains
    multiple bracket accessors.

    Parameters
    ----------
    string: str
        The string to inspect

    Returns
    -------
    str
        The name of the variable in the string.
    str
        The index that was inside of the accessor brackets.
    """
    if string.count('[') > 1 or string.count(']') > 1:
        raise ValueError('Nested array indices (e.g. values[0][0]) are not '
                         'supported: {}'.format(string))

    start_bracket_index = string.find('[')
    end_bracket_index = string.find(']')

    if start_bracket_index >= 0 > end_bracket_index:
        raise ValueError('Property name containts a [ without a matching ]: '
                         '{}'.format('.'.join(string)))

    if end_bracket_index >= 0 > start_bracket_index:
        raise ValueError('Property name containts a ] without a matching [: '
                         '{}'.format(string))

    if end_bracket_index == start_bracket_index + 1:
        raise ValueError('There is no index between the array brackets: '
                         '{}'.format(string))

    if end_bracket_index != len(string) - 1:
        raise ValueError('The ] array bracket must be at the end of the property name: '
                         '{}'.format(string))

    array_index = string[start_bracket_index + 1: end_bracket_index]
    property_name = string[0: start_bracket_index]

    return property_name, array_index
