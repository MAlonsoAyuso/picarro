from dataclasses import dataclass
import datetime
import pandas as pd
import numpy as np
import scipy.stats


@dataclass
class LinearFit:
    start_time: datetime.datetime
    end_time: datetime.datetime
    intercept: np.float64
    slope: np.float64
    intercept_stderr: np.float64
    slope_stderr: np.float64

    def predict(self, times):
        elapsed_time_as_num = time_to_num(times, self.start_time)
        values = self.intercept + self.slope * elapsed_time_as_num
        return pd.Series(data=values, index=times)

    def predict_endpoints(self):
        times = pd.DatetimeIndex([self.start_time, self.end_time])
        return self.predict(times)


def time_to_num(time, start_time):
    return (time - start_time).total_seconds()


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

def estimate_vol_flux(measurement, t0, skip_start, h, tau, conc_unit_prefix):
    linear_fit = fit_line(measurement, skip_start, skip_end=0)
    fit_duration = (linear_fit.end_time - linear_fit.start_time).total_seconds()
    t_mid = skip_start + fit_duration / 2
    vol_flux = h * np.exp(t_mid / tau) * linear_fit.slope * conc_unit_prefix
    return vol_flux
