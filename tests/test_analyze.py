import pandas as pd
from picarro.chunks import _read_file, PicarroColumns
from picarro.analyze import FluxEstimationConfig, estimate_flux
import pathlib
import numpy as np

from picarro.config import MeasurementsConfig

_DATA_DIR = pathlib.Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


def abs_rel_diff(a, b):
    return np.abs((a - b) / b)


common_params = dict(
    t0_delay=pd.Timedelta(8 * 60, "s"),
    t0_margin=pd.Timedelta(2 * 60, "s"),
    A=0.25,  # m2
    Q=0.25 * 1e-3 / 60,  # m3/s
    V=50e-3,  # m3
    columns=[],  # does not matter here
)

linear_config = FluxEstimationConfig(
    method="linear",
    **common_params,  # type: ignore
)

exponential_config = FluxEstimationConfig(
    method="exponential",
    **common_params,  # type: ignore
)

measurement_config = MeasurementsConfig(
    valve_column=PicarroColumns.solenoid_valves,
    src=str(data_path("example_measurement.dat")),
    columns=["N2O", "CO2"],
)


def test_linear_N2O_slope_right_order_of_magnitude():
    # This test will catch any serious errors in order of magnitude etc
    measurement = _read_file(data_path("example_measurement.dat"), measurement_config)[
        PicarroColumns.N2O
    ]
    linear_estimator = estimate_flux(linear_config, measurement)
    linear_fit = linear_estimator.fit_params
    assert abs_rel_diff(linear_fit.slope, 1.44e-4) < 0.01  # 1.44 * 10^(-4) ppmv / s


def test_fit_line_approximates_values():
    # Using CO2 here because it has low noise and thus it will be very clear
    # if the values do not fit.
    measurement = _read_file(data_path("example_measurement.dat"), measurement_config)[
        PicarroColumns.CO2
    ]

    estimator = estimate_flux(linear_config, measurement)

    assert isinstance(measurement.index, pd.DatetimeIndex)  # type: ignore
    times = measurement.index  # type: ignore
    times_in_regression = times[
        (times >= estimator.moments.fit_start) & (times <= estimator.moments.fit_end)
    ]
    num_points_to_predict = len(times_in_regression)
    assert num_points_to_predict == 741

    predictions = estimator.predict(times_in_regression)
    rel_diff = ((predictions - measurement) / measurement).dropna()

    assert len(rel_diff) == num_points_to_predict

    assert rel_diff.mean() <= 0.003  # average relative difference << 1% for CO2

    # about equal shares of the predictions should be above and below
    assert (rel_diff < 0).mean() > 0.45
    assert (rel_diff > 0).mean() > 0.45

    # large (here 10%) deviations should be uncommon (here 5% of each tail)
    assert rel_diff.quantile(0.05) >= -0.10
    assert rel_diff.quantile(0.95) <= 0.10


def test_estimate_N2O_vol_flux_right_order_of_magnitude():
    measurement = _read_file(data_path("example_measurement.dat"), measurement_config)[
        PicarroColumns.N2O
    ]
    estimator = estimate_flux(linear_config, measurement)
    vol_flux = estimator.estimate_vol_flux()

    slope = 1.44022e-4  # ppmv / s
    moments = estimator.moments
    elapsed_time_at_mid_of_fit = (
        moments.fit_start + (moments.fit_end - moments.fit_start) / 2 - moments.t0
    ).total_seconds()
    correction_factor = np.exp(elapsed_time_at_mid_of_fit / linear_config.tau)
    expected_result = linear_config.h * slope * correction_factor

    assert abs_rel_diff(vol_flux, expected_result) < 0.001  # m/s


def test_estimate_N2O_vol_flux_linear_and_exponential_agree():
    measurement = _read_file(data_path("example_measurement.dat"), measurement_config)[
        PicarroColumns.N2O
    ]
    linear_estimator = estimate_flux(linear_config, measurement)
    exponential_estimator = estimate_flux(exponential_config, measurement)

    vol_flux_linear = linear_estimator.estimate_vol_flux()
    vol_flux_exponential = exponential_estimator.estimate_vol_flux()

    assert abs_rel_diff(vol_flux_linear, vol_flux_exponential) < 0.001  # m/s
