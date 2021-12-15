import pytest
from picarro.read import read_raw, PicarroColumns
from picarro.analyze import fit_line
import pathlib
import numpy as np

_DATA_DIR = pathlib.Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


def same_order_of_magnitude(value, reference_value):
    return np.round(value / reference_value) == 1


def test_N2O_slope_right_order_of_magnitude():
    measurement = read_raw(data_path("example_measurement.dat"))[PicarroColumns.N2O]
    linear_fit = fit_line(measurement, skip_start=60 * 10)
    assert same_order_of_magnitude(linear_fit.slope, 1e-4)  # 10^(-4) ppmv / s


def test_fit_line_approximates_values():
    # Using CO2 here because it has low noise and thus it will be very clear
    # if the values do not fit.
    measurement = read_raw(data_path("example_measurement.dat"))[PicarroColumns.CO2]

    linear_fit = fit_line(measurement, skip_start=60 * 10)

    times = measurement.index
    times_in_regression = times[
        (times >= linear_fit.start_time) & (times <= linear_fit.end_time)
    ]
    num_points_to_predict = len(times_in_regression)
    assert num_points_to_predict == 741

    predictions = linear_fit.predict(times_in_regression)
    rel_diff = ((predictions - measurement) / measurement).dropna()

    assert len(rel_diff) == num_points_to_predict

    assert rel_diff.mean() <= 0.003  # average relative difference << 1% for CO2

    # about equal shares of the predictions should be above and below
    assert (rel_diff < 0).mean() > 0.45
    assert (rel_diff > 0).mean() > 0.45

    # large (here 10%) deviations should be uncommon (here 5% of each tail)
    assert rel_diff.quantile(0.05) >= -0.10
    assert rel_diff.quantile(0.95) <= 0.10
