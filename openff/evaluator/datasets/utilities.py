from typing import TYPE_CHECKING, Set, Tuple

if TYPE_CHECKING:
    import pandas


def reorder_data_frame(data_frame: "pandas.DataFrame") -> "pandas.DataFrame":
    """Re-order the substance columns of a data frame so that the individual
    components are alphabetically sorted.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame to re-order.

    Returns
    -------
    pandas.DataFrame
        The re-ordered data frame.
    """

    import numpy
    import pandas

    min_n_components = data_frame["N Components"].min()
    max_n_components = data_frame["N Components"].max()

    if max_n_components > 2:
        raise NotImplementedError(
            "Reordering more than 2 components has not yet been robustly tested."
        )

    ordered_frames = []

    for n_components in range(min_n_components, max_n_components + 1):
        component_frame = data_frame[data_frame["N Components"] == n_components]
        ordered_frame = data_frame[data_frame["N Components"] == n_components].copy()

        component_headers = [f"Component {i + 1}" for i in range(n_components)]

        # this was a dataframe in pandas 1, but an array in pandas 2
        component_order = numpy.argsort(component_frame[component_headers], axis=1)

        substance_headers = ["Component", "Role", "Mole Fraction", "Exact Amount"]

        for component_index in range(n_components):
            indices = component_order[:, component_index]

            for substance_header in substance_headers:
                component_header = f"{substance_header} {component_index + 1}"

                for replacement_index in range(n_components):
                    if component_index == replacement_index:
                        continue

                    replacement_header = f"{substance_header} {replacement_index + 1}"

                    ordered_frame[component_header] = numpy.where(
                        indices == replacement_index,
                        (
                            numpy.nan
                            if replacement_header not in component_frame
                            else component_frame[replacement_header]
                        ),
                        (
                            numpy.nan
                            if component_header not in component_frame
                            else component_frame[component_header]
                        ),
                    )

        ordered_frames.append(ordered_frame)

    ordered_data_frame = pandas.concat(ordered_frames, ignore_index=True, sort=False)

    return ordered_data_frame


def data_frame_to_substances(data_frame: "pandas.DataFrame") -> Set[Tuple[str, ...]]:
    """Extracts all unique substances from a data frame and returns them
    as a set, where each element in the set is a tuple of smiles patterns
    which represent a single substance.

    Parameters
    ----------
    data_frame
        The data frame to extract the substances from.

    Returns
    -------
        The set of unique substances.
    """

    if len(data_frame) == 0:
        return set()

    ordered_data = reorder_data_frame(data_frame)

    substances: Set[Tuple[str, ...]] = set()

    min_n_components = data_frame["N Components"].min()
    max_n_components = data_frame["N Components"].max()

    for n_components in range(min_n_components, max_n_components + 1):
        component_data = ordered_data[ordered_data["N Components"] == n_components]

        component_columns = (
            component_data[f"Component {i + 1}"] for i in range(n_components)
        )

        substances.update(list(zip(*component_columns)))

    return substances
