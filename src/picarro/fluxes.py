from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats

from picarro.util import format_duration

logger = logging.getLogger(__name__)

VolumetricFlux = float


@dataclass
class EstimationParams:
    t0_delay: datetime.timedelta  # the nominal delay from valve switch to gas arrival
    t0_margin: datetime.timedelta  # an additional delay to add before starting fit
    tau: datetime.timedelta  # time constant (= V / Q)
    h: float  # equivalent height of chamber (= V / A)


@dataclass
class LinearFit:
    intercept: np.float64
    slope: np.float64
    intercept_stderr: np.float64
    slope_stderr: np.float64


@dataclass
class ExponentialEstimator:
    gas: str
    moments: Moments
    params: EstimationParams
    linear_fit: LinearFit

    @classmethod
    def fit(cls, data: pd.Series, params: EstimationParams) -> ExponentialEstimator:
        if not isinstance(data.name, str):
            raise ValueError(f"Expected string name on series but found {data.name}.")

        moments = Moments.from_data(data, params)

        data_to_fit = data[moments.fit_start : moments.fit_end]
        assert isinstance(data_to_fit.index, pd.DatetimeIndex)
        assert len(data_to_fit), (data, moments)

        x = cls.transform_time(data_to_fit.index, moments.t0, params.tau)
        y = data_to_fit.to_numpy()  # type: ignore
        scipy_fit_result = scipy.stats.linregress(x, y)
        linear_fit = LinearFit(
            intercept=scipy_fit_result.intercept,  # type: ignore
            slope=scipy_fit_result.slope,  # type: ignore
            intercept_stderr=scipy_fit_result.intercept_stderr,  # type: ignore
            slope_stderr=scipy_fit_result.stderr,  # type: ignore
        )

        return cls(
            data.name,
            moments,
            params,
            linear_fit,
        )

    @staticmethod
    def transform_time(
        times: pd.DatetimeIndex, t0: datetime.datetime, tau: datetime.timedelta
    ) -> np.ndarray:
        return 1 - np.exp(-calculate_elapsed_seconds(times, t0) / tau.total_seconds())

    def estimate_vol_flux(self) -> VolumetricFlux:
        vol_flux = (
            self.params.h / self.params.tau.total_seconds() * self.linear_fit.slope
        )
        return float(vol_flux)

    def predict_concentration(self, times: pd.DatetimeIndex) -> pd.Series:
        x = self.transform_time(times, self.moments.t0, self.params.tau)
        y = self.linear_fit.intercept + self.linear_fit.slope * x
        return pd.Series(data=y, index=times)


@dataclass
class Moments:
    data_start: datetime.datetime
    t0: datetime.datetime
    fit_start: datetime.datetime
    fit_end: datetime.datetime
    data_end: datetime.datetime

    @staticmethod
    def from_data(data: pd.Series, params: EstimationParams) -> Moments:
        assert isinstance(data.index, pd.DatetimeIndex)
        if len(data) == 0:
            raise ValueError(f"Empty dataset {data}")
        data_start = data.index[0]
        data_end = data.index[-1]
        t0 = data_start + params.t0_delay
        fit_start_limit = t0 + params.t0_margin
        fit_end_limit = data_end

        if fit_start_limit > data_end:
            data_duration = data_end - data_start
            raise ValueError(
                f"Check settings. "
                f"Measurement starting at {data_start} "
                f"has duration {format_duration(data_duration)} "
                f"and is therefore too short to start the fit after "
                f"{format_duration(params.t0_delay + params.t0_margin)}."
            )

        fit_index = data.index[
            (data.index >= fit_start_limit) & (data.index <= fit_end_limit)
        ]
        if len(fit_index) == 0:
            raise ValueError(
                f"Check limits! No data in fit range {[fit_start_limit, fit_end_limit]} "
                f"for dataset \n{data}"
            )
        fit_start, fit_end = fit_index[0], fit_index[-1]

        return Moments(data_start, t0, fit_start, fit_end, data_end)


def calculate_elapsed_seconds(
    times: pd.DatetimeIndex, ref_time: datetime.datetime
) -> np.ndarray:
    return (times - ref_time).total_seconds()  # type: ignore
