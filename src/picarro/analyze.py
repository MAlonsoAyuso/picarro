from __future__ import annotations
from typing import Mapping, Type, Union
from dataclasses import dataclass
import datetime
import pandas as pd
import numpy as np
import scipy.stats
from picarro.config import FluxEstimationConfig

VolumetricFlux = float

TimeSeries = pd.Series


@dataclass
class LinearFit:
    intercept: np.float64
    slope: np.float64
    intercept_stderr: np.float64
    slope_stderr: np.float64


@dataclass
class Moments:
    data_start: datetime.datetime
    t0: datetime.datetime
    fit_start: datetime.datetime
    fit_end: datetime.datetime
    data_end: datetime.datetime


@dataclass
class _FluxEstimatorBase:
    config: FluxEstimationConfig
    column: str
    fit_params: LinearFit
    moments: Moments

    @staticmethod
    def transform_time(
        times: pd.DatetimeIndex, config: FluxEstimationConfig, moments: Moments
    ) -> np.ndarray:
        raise NotImplementedError()

    def estimate_vol_flux(self) -> VolumetricFlux:
        raise NotImplementedError()

    def predict(self, times: pd.DatetimeIndex) -> TimeSeries:
        x = self.transform_time(times, self.config, self.moments)
        y = self.fit_params.intercept + self.fit_params.slope * x
        return pd.Series(data=y, index=times)

    @staticmethod
    def _calculate_elapsed(
        times: pd.DatetimeIndex, t0: datetime.datetime
    ) -> pd.TimedeltaIndex:
        elapsed = times - t0
        assert isinstance(elapsed, pd.TimedeltaIndex)
        return elapsed

    @classmethod
    def create(cls, data: TimeSeries, config: FluxEstimationConfig):
        column = data.name
        if not column in config.volume_prefixes:
            raise ValueError(
                f"missing column {column} in volume_prefixes {config.volume_prefixes}"
            )
        assert isinstance(column, str)
        fit_params, moments = cls._fit(data, config)
        assert isinstance(data.index, pd.DatetimeIndex)
        return cls(config, column, fit_params, moments)

    @classmethod
    def _fit(
        cls, data: TimeSeries, config: FluxEstimationConfig
    ) -> tuple[LinearFit, Moments]:
        moments = cls._determine_moments(data, config)
        data_to_fit = data[moments.fit_start : moments.fit_end]
        assert isinstance(data_to_fit.index, pd.DatetimeIndex)
        assert len(data), (data, moments)
        x = cls.transform_time(data_to_fit.index, config, moments)
        y = data_to_fit.to_numpy()
        result = scipy.stats.linregress(x, y)
        fit_params = LinearFit(
            intercept=result.intercept,  # type: ignore
            slope=result.slope,  # type: ignore
            intercept_stderr=result.intercept_stderr,  # type: ignore
            slope_stderr=result.stderr,  # type: ignore
        )
        return fit_params, moments

    @staticmethod
    def _determine_moments(data: TimeSeries, config: FluxEstimationConfig) -> Moments:
        assert isinstance(data.index, pd.DatetimeIndex)
        if len(data) == 0:
            raise ValueError(f"Empty dataset {data}")
        data_start = data.index[0]
        data_end = data.index[-1]
        t0 = data_start + datetime.timedelta(seconds=config.t0_delay)
        fit_start_limit = t0 + datetime.timedelta(seconds=config.t0_margin)
        fit_end_limit = data_end - datetime.timedelta(seconds=config.skip_end)

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


@dataclass
class LinearEstimator(_FluxEstimatorBase):
    @staticmethod
    def transform_time(
        times: pd.DatetimeIndex, config: FluxEstimationConfig, moments: Moments
    ) -> np.ndarray:
        # Since estimation is linear, it does not matter what start point we have.
        # Referencing here from t0 for easier debugging: now we know
        # that the regression x variable represents elapsed seconds since t0.
        return calculate_elapsed_seconds(times, moments.t0)

    def estimate_vol_flux(self) -> VolumetricFlux:
        # In principle we want to estimate the derivative of the exponential curve
        # at a given time in relation to t0.
        # But in practice the fit here is a linear fit to a part of the curve.
        # Here I'm assuming that the slope of the linear fit is approximately
        # equal to the derivative at the midpoint of the exponential fit.
        fit_duration = self.moments.fit_end - self.moments.fit_start
        fit_midpoint = self.moments.fit_start + fit_duration / 2
        seconds_elapsed = (fit_midpoint - self.moments.t0).total_seconds()
        h = self.config.V / self.config.A
        tau = self.config.V / self.config.Q
        volume_prefix = self.config.volume_prefixes[self.column]
        vol_flux = (
            h * np.exp(seconds_elapsed / tau) * self.fit_params.slope * volume_prefix
        )
        return vol_flux


@dataclass
class ExponentialEstimator(_FluxEstimatorBase):
    @classmethod
    def transform_time(
        cls, times: pd.DatetimeIndex, config: FluxEstimationConfig, moments: Moments
    ) -> np.ndarray:
        # Since estimation is linear, it does not matter what unit we have for time.
        # Referencing here from times[0] for easier debugging: now we know
        # that the regression x variable represents elapsed seconds of the measurement.
        tau = config.V / config.Q
        elapsed_seconds = (times - moments.t0).total_seconds()  # type: ignore
        return 1 - np.exp(-elapsed_seconds / tau)

    def estimate_vol_flux(self) -> VolumetricFlux:
        # In principle we want to estimate the derivative of the exponential curve
        # at a given time in relation to t0.
        # But in practice the fit here is a linear fit to a part of the curve.
        # Here I'm assuming that the slope of the linear fit is approximately
        # equal to the derivative at the midpoint of the exponential fit.
        h = self.config.V / self.config.A
        tau = self.config.V / self.config.Q
        volume_prefix = self.config.volume_prefixes[self.column]
        vol_flux = h / tau * self.fit_params.slope * volume_prefix
        return float(vol_flux)


def calculate_elapsed_seconds(
    times: pd.DatetimeIndex, ref_time: datetime.datetime
) -> np.ndarray:
    return (times - ref_time).total_seconds()  # type: ignore


FluxEstimator = Union[LinearEstimator, ExponentialEstimator]

_ESTIMATORS: Mapping[str, Type[FluxEstimator]] = {
    "linear": LinearEstimator,
    "exponential": ExponentialEstimator,
}


def estimate_flux(config: FluxEstimationConfig, data: TimeSeries) -> FluxEstimator:
    cls = _ESTIMATORS[config.method]
    return cls.create(data, config)
