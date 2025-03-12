"""
A set of utilities for performing statistical analysis on a time series.

Notes
-----
Based on the original implementation found in `pymbar
<https://github.com/choderalab/pymbar/tree/master/pymbar>`_
"""

import math
from typing import List

import numpy as np
from pymbar.utils import ParameterError

from openff.evaluator.attributes import Attribute, AttributeClass


class TimeSeriesStatistics(AttributeClass):
    """A class which encodes statistics such as the statistical inefficiency and
    the index after which the time series has become stationary (i.e. is equilibrated).
    """

    n_total_points: int = Attribute(
        docstring="The total number of data point in the time series.", type_hint=int
    )
    n_uncorrelated_points: int = Attribute(
        docstring="The number of data point in the time series which are "
        "uncorrelated.",
        type_hint=int,
    )

    statistical_inefficiency: float = Attribute(
        docstring="The statistical inefficiency of the time series.", type_hint=float
    )
    equilibration_index: int = Attribute(
        docstring="The index after which the time series has become stationary.",
        type_hint=int,
    )

    def __init__(
        self,
        n_total_points: int = None,
        n_uncorrelated_points: int = None,
        statistical_inefficiency: float = None,
        equilibration_index: int = None,
    ):
        if n_total_points is not None:
            self.n_total_points = n_total_points
        if n_uncorrelated_points is not None:
            self.n_uncorrelated_points = n_uncorrelated_points
        if statistical_inefficiency is not None:
            self.statistical_inefficiency = statistical_inefficiency
        if equilibration_index is not None:
            self.equilibration_index = equilibration_index


def _statistical_inefficiency(
    time_series: np.ndarray, minimum_samples: int = 3
) -> float:
    """Calculates the statistical inefficiency of a time series.

    Parameters
    ----------
    time_series
        The time series to analyse with shape=(n_data_points, n_dimensions).
    minimum_samples: int
        The minimum number of data points to consider in the calculation.

    Notes
    -----
    This method is based on the paper by J. D. Chodera [1]_ and the implementation at
    https://github.com/choderalab/pymbar. Here the code is extended support
    multidimensional data such as dipole moments.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the
        weighted histogram analysis method for the analysis of simulated and parallel
        tempering simulations. JCTC 3(1):26-41, 2007.

    Returns
    -------
        The statistical inefficiency.
    """

    # Make sure the time series has a consistent shape of (n_data_points, n_dimensions)
    dimension = 1 if len(time_series.shape) == 1 else time_series.shape[1]
    standardised_time_series = time_series.reshape((len(time_series), dimension))

    number_of_timesteps = standardised_time_series.shape[0]

    time_series_mean = standardised_time_series.mean(axis=0)
    time_series_shifted = standardised_time_series - time_series_mean

    sigma_squared = np.mean(
        np.sum(time_series_shifted * time_series_shifted, axis=1), axis=0
    )

    if sigma_squared == 0:
        raise ParameterError(
            "Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency"
        )

    current_timestep = 1
    statistical_inefficiency = 1.0

    while current_timestep < number_of_timesteps - 1:
        autocorrelation_function = np.sum(
            np.sum(
                time_series_shifted[0 : (number_of_timesteps - current_timestep)]
                * time_series_shifted[current_timestep:number_of_timesteps],
                axis=1,
            ),
            axis=0,
        ) / (float(number_of_timesteps - current_timestep) * sigma_squared)

        if autocorrelation_function <= 0.0 and current_timestep > minimum_samples:
            break

        statistical_inefficiency += (
            2.0
            * autocorrelation_function
            * (1.0 - float(current_timestep) / float(number_of_timesteps))
        )

        current_timestep += 1

    # Enforce a minimum autocorrelation time of 0.
    if statistical_inefficiency < 1.0:
        statistical_inefficiency = 1.0

    return statistical_inefficiency


def analyze_time_series(
    time_series: np.ndarray,
    discard_initial_frames: int = 0,
    minimum_samples: int = 3
) -> TimeSeriesStatistics:
    """Detect when a time series set has effectively become stationary (i.e has reached
    equilibrium).

    Parameters
    ----------
    time_series
        The time series to analyse with shape=(n_data_points, n_dimensions).
    discard_initial_frames
        The number of initial frames to discard.
    minimum_samples
        The minimum number of data points to consider in the calculation.

    Notes
    -----
    This method is based on the paper by J. D. Chodera [1]_ and the implementation at
    https://github.com/choderalab/pymbar. Here the code is extended support
    multidimensional data such as dipole moments.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the
        weighted histogram analysis method for the analysis of simulated and parallel
        tempering simulations. JCTC 3(1):26-41, 2007.

    Returns
    -------
        Statistics about the time series.
    """

    n_timesteps = time_series.shape[0]
    statistical_inefficiency_array = np.ones([n_timesteps - 1])

    # Special case if the time series is constant.
    if np.isclose(time_series.std(), 0.0):
        return TimeSeriesStatistics(
            n_total_points=len(time_series),
            n_uncorrelated_points=1,
            statistical_inefficiency=float(len(time_series)),
            equilibration_index=0,
        )

    effect_samples_array = np.ones([n_timesteps - 1])

    for current_timestep in range(discard_initial_frames, n_timesteps - 1):
        try:
            statistical_inefficiency_array[current_timestep] = (
                _statistical_inefficiency(
                    time_series[current_timestep:n_timesteps], minimum_samples
                )
            )

        except ParameterError:
            # Fix for issue https://github.com/choderalab/pymbar/issues/122
            statistical_inefficiency_array[current_timestep] = (
                n_timesteps - current_timestep + 1
            )

        effect_samples_array[current_timestep] = (
            n_timesteps - current_timestep + 1
        ) / statistical_inefficiency_array[current_timestep]

    equilibration_time = effect_samples_array.argmax()
    statistical_inefficiency = statistical_inefficiency_array[equilibration_time]

    return TimeSeriesStatistics(
        n_total_points=len(time_series),
        n_uncorrelated_points=len(
            get_uncorrelated_indices(
                len(time_series),
                statistical_inefficiency,
                discard_initial_frames
            )
        ),
        statistical_inefficiency=float(statistical_inefficiency),
        equilibration_index=int(equilibration_time),
    )


def get_uncorrelated_indices(
    time_series_length: int,
    statistical_inefficiency: float,
    discard_initial_frames: int = 0,
) -> List[int]:
    """Returns the indices of the uncorrelated frames of a time series taking strides
    computed by the ``get_uncorrelated_stride`` function.

    Parameters
    ----------
    time_series_length
        The length of the time series to extract frames from.
    statistical_inefficiency
        The statistical inefficiency of the time series.
    discard_initial_frames
        The number of initial frames to discard. Samples will be taken from
        ``discard_initial_frames`` to ``time_series_length``.

    Returns
    -------
        The indices of the uncorrelated frames.
    """

    # Extract a set of uncorrelated data points.
    stride = int(math.ceil(statistical_inefficiency))
    return [
        index
        for index in range(discard_initial_frames, time_series_length, stride)
    ]
