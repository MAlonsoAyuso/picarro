from __future__ import annotations
from typing import List, Mapping, Type, Union
from dataclasses import dataclass, field
import datetime
import pandas as pd
import numpy as np
import scipy.stats
import logging
import cattr.preconf.json

logger = logging.getLogger(__name__)

VolumetricFlux = float


@dataclass
class FluxesConfig:
    t0_delay: datetime.timedelta
    t0_margin: datetime.timedelta
    A: float
    V: float
    Q: float
    columns: List[str] = field(default_factory=list)
    method: str = "exponential"

    @property
    def tau(self) -> float:
        return self.V / self.Q

    @property
    def h(self) -> float:
        return self.V / self.A

    def __post_init__(self):
        if self.method not in ESTIMATORS:
            raise ValueError(f"{self.method!r} is not a flux estimation method.")


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
    config: FluxesConfig
    column: str
    fit_params: LinearFit
    moments: Moments
    n_samples: int

    def unstructure(self) -> dict:
        return json_converter.unstructure(self)

    @staticmethod
    def transform_time(
        times: pd.DatetimeIndex, config: FluxesConfig, moments: Moments
    ) -> np.ndarray:
        raise NotImplementedError()

    def estimate_vol_flux(self) -> VolumetricFlux:
        raise NotImplementedError()

    def predict(self, times: pd.DatetimeIndex) -> pd.Series:
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
    def create(cls, data: pd.Series, config: FluxesConfig):
        column = data.name
        assert isinstance(column, str)
        fit_params, moments, n_samples = cls._fit(data, config)
        assert isinstance(data.index, pd.DatetimeIndex)
        return cls(config, column, fit_params, moments, n_samples)

    @classmethod
    def _fit(
        cls, data: pd.Series, config: FluxesConfig
    ) -> tuple[LinearFit, Moments, int]:
        moments = cls._determine_moments(data, config)
        data_to_fit = data[moments.fit_start : moments.fit_end]
        assert isinstance(data_to_fit.index, pd.DatetimeIndex)
        assert len(data_to_fit), (data, moments)
        x = cls.transform_time(data_to_fit.index, config, moments)
        y = data_to_fit.to_numpy()  # type: ignore
        result = scipy.stats.linregress(x, y)
        fit_params = LinearFit(
            intercept=result.intercept,  # type: ignore
            slope=result.slope,  # type: ignore
            intercept_stderr=result.intercept_stderr,  # type: ignore
            slope_stderr=result.stderr,  # type: ignore
        )
        return fit_params, moments, len(data_to_fit)

    @staticmethod
    def _determine_moments(data: pd.Series, config: FluxesConfig) -> Moments:
        assert isinstance(data.index, pd.DatetimeIndex)
        if len(data) == 0:
            raise ValueError(f"Empty dataset {data}")
        data_start = data.index[0]
        data_end = data.index[-1]
        t0 = data_start + config.t0_delay
        fit_start_limit = t0 + config.t0_margin
        fit_end_limit = data_end

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
        times: pd.DatetimeIndex, config: FluxesConfig, moments: Moments
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
        vol_flux = (
            self.config.h
            * np.exp(seconds_elapsed / self.config.tau)
            * self.fit_params.slope
        )
        return vol_flux


@dataclass
class ExponentialEstimator(_FluxEstimatorBase):
    @staticmethod
    def transform_time(
        times: pd.DatetimeIndex, config: FluxesConfig, moments: Moments
    ) -> np.ndarray:
        # Since estimation is linear, it does not matter what unit we have for time.
        # Referencing here from times[0] for easier debugging: now we know
        # that the regression x variable represents elapsed seconds of the measurement.
        elapsed_seconds = (times - moments.t0).total_seconds()  # type: ignore
        return 1 - np.exp(-elapsed_seconds / config.tau)

    def estimate_vol_flux(self) -> VolumetricFlux:
        # In principle we want to estimate the derivative of the exponential curve
        # at a given time in relation to t0.
        # But in practice the fit here is a linear fit to a part of the curve.
        # Here I'm assuming that the slope of the linear fit is approximately
        # equal to the derivative at the midpoint of the exponential fit.
        vol_flux = self.config.h / self.config.tau * self.fit_params.slope
        return float(vol_flux)


def calculate_elapsed_seconds(
    times: pd.DatetimeIndex, ref_time: datetime.datetime
) -> np.ndarray:
    return (times - ref_time).total_seconds()  # type: ignore


FluxEstimator = Union[LinearEstimator, ExponentialEstimator]

ESTIMATORS: Mapping[str, Type[FluxEstimator]] = {
    "linear": LinearEstimator,
    "exponential": ExponentialEstimator,
}


# @dataclass
# class FluxResult:
#     segment_info: SegmentInfo
#     estimator: FluxEstimator


def estimate_flux(config: FluxesConfig, data: pd.Series) -> FluxEstimator:
    cls = ESTIMATORS[config.method]
    return cls.create(data, config)


def build_fluxes_dataframe(flux_results: List[FluxResult]) -> pd.DataFrame:
    def make_row(flux_result: FluxResult):
        return pd.Series(
            dict(
                start_utc=flux_result.measurement_meta.start,
                end_utc=flux_result.measurement_meta.end,
                valve_number=flux_result.measurement_meta.valve_number,
                valve_label=flux_result.measurement_meta.valve_label,
                column=flux_result.estimator.column,
                vol_flux=flux_result.estimator.estimate_vol_flux(),
                n_samples_total=flux_result.measurement_meta.n_samples,
                n_samples_flux_estimate=flux_result.estimator.n_samples,
            )
        )

    rows = list(map(make_row, flux_results))
    return pd.DataFrame(rows)


json_converter = cattr.preconf.json.make_converter()
json_converter.register_unstructure_hook(datetime.datetime, datetime.datetime.isoformat)
json_converter.register_unstructure_hook(
    datetime.timedelta, datetime.timedelta.total_seconds
)
json_converter.register_structure_hook(
    datetime.datetime, lambda v, _: datetime.datetime.fromisoformat(v)
)
json_converter.register_structure_hook(
    datetime.timedelta, lambda v, _: datetime.timedelta(seconds=v)
)
json_converter.register_structure_hook(
    FluxEstimator,
    lambda obj, _: json_converter.structure(obj, ESTIMATORS[obj["config"]["method"]]),
)
