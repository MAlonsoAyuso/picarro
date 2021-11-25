from dataclasses import dataclass
import datetime
from os import times
import pandas as pd
import scipy.stats


@dataclass
class LinearFit:
    start_time: datetime.datetime
    end_time: datetime.datetime
    intercept: float
    slope: float
    intercept_stderr: float
    slope_stderr: float


def time_to_num(time, start_time):
    return (time - start_time).seconds


def predict_two_points(linear_fit):
    times = pd.DatetimeIndex([linear_fit.start_time, linear_fit.end_time])
    elapsed_time_as_num = time_to_num(times, linear_fit.start_time)
    values = linear_fit.intercept + linear_fit.slope * elapsed_time_as_num
    return pd.Series(data=values, index=times)


def fit_line(data, skip_start=0, skip_end=0):
    data_start = data.index[0]
    data_end = data.index[-1]
    subset_start_limit = data_start + datetime.timedelta(seconds=skip_start)
    subset_end_limit = data_end - datetime.timedelta(seconds=skip_end)
    subset = data[(data.index >= subset_start_limit) & (data.index <= subset_end_limit)]
    subset_start = subset.index[0]
    subset_end = subset.index[-1]

    result = scipy.stats.linregress(
        time_to_num(subset.index, subset_start), subset.values
    )

    return LinearFit(
        start_time=subset_start,
        end_time=subset_end,
        intercept=result.intercept,
        slope=result.slope,
        intercept_stderr=result.intercept_stderr,
        slope_stderr=result.stderr,
    )
