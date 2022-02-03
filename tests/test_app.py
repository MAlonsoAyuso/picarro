import os
from pathlib import Path
import shutil
import itertools
from typing import Callable
import pytest
import picarro.app
from picarro.config import (
    AppConfig,
    FluxEstimationConfig,
    MeasurementsConfig,
    OutItem,
)
from picarro.measurements import MeasurementMeta
import picarro.measurements
import numpy as np
import pandas as pd


config_example_src = Path(__file__).absolute().parent / "config_example.toml"
assert config_example_src.exists()

_EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"


def abs_rel_diff(a, b):
    return np.abs((a - b) / b)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config_path = tmp_path / "config-filename-stem.toml"
    shutil.copyfile(config_example_src, config_path)
    os.chdir(config_path.parent)
    return AppConfig.from_toml(config_path)


def test_create_config(app_config: AppConfig, tmp_path: Path):
    expected_conf = AppConfig(
        measurements=MeasurementsConfig(
            valve_column="solenoid_valves",
            src="data-dir/**/*.dat",
            extra_columns=["N2O", "CH4", "CO2"],
            max_gap=pd.Timedelta(5, "s"),
            min_duration=pd.Timedelta(1080, "s"),
            max_duration=None,
        ),
        flux_estimation=FluxEstimationConfig(
            columns=["N2O", "CH4"],
            method="linear",
            t0_delay=pd.Timedelta(480, "s"),
            t0_margin=pd.Timedelta(120, "s"),
            A=0.25,
            Q=4.16e-6,
            V=50e-3,
        ),
    )

    assert app_config == expected_conf


def call_immediately(func: Callable[[], None]) -> None:
    func()


def summarize_measurement(mm: MeasurementMeta):
    return dict(valve_number=mm.valve_number, n_samples=mm.n_samples)


def test_integrated(app_config: AppConfig, tmp_path: Path):
    data_dir = tmp_path / "data-dir" / "nested" / "path"
    shutil.copytree(_EXAMPLE_DATA_DIR / "adjacent_files", data_dir)

    @call_immediately
    def test_will_not_overwrite():  # pyright: reportUnusedFunction=false
        # Test that it won't do anything if the output path already exists
        out_path = app_config.output.get_path(OutItem.measurement_metas_json)
        out_dir = out_path.parent
        out_dir.mkdir()
        out_path.touch()
        with pytest.raises(picarro.app.PicarroPathExists):
            picarro.app.identify_and_save_measurement_metas(app_config)
        os.remove(out_path)

    # These were established by manually sifting through the files
    expected_summaries = [
        # one removed here compared to the full set, because it's too short
        dict(valve_number=14, n_samples=1789),
        dict(valve_number=15, n_samples=1787),
        dict(valve_number=1, n_samples=1779),
        dict(valve_number=2, n_samples=1782),
        dict(valve_number=3, n_samples=1789),
        dict(valve_number=4, n_samples=1786),
        dict(valve_number=5, n_samples=1783),
        # one removed here compared to the full set, because it's too short
    ]

    picarro.app.identify_and_save_measurement_metas(app_config)

    @call_immediately
    def test_measurement_metadata():
        # Check that measurement metadata objects are as expected
        measurement_summaries = [
            summarize_measurement(mm)
            for mm in picarro.app.load_measurement_metas(app_config)
        ]

        assert measurement_summaries == expected_summaries

    @call_immediately
    def test_measurement_data():
        # Check that measurement datasets are as expected
        measurements = picarro.measurements.read_measurements(
            picarro.app.load_measurement_metas(app_config),
            app_config.measurements,
        )
        data_summaries = [
            dict(
                valve_number=m[app_config.measurements.valve_column].unique()[0],
                n_samples=len(m),
            )
            for m in measurements
        ]

        assert data_summaries == expected_summaries

    @call_immediately
    def test_flux_estimation_working():
        # Test flux estimation
        assert app_config.flux_estimation
        flux_results = list(picarro.app.estimate_fluxes(app_config))
        expected_flux_results = list(
            itertools.product(expected_summaries, app_config.flux_estimation.columns)
        )
        seen_flux_results = [
            (summarize_measurement(ar.measurement_meta), ar.estimator.column)
            for ar in flux_results
        ]
        assert expected_flux_results == seen_flux_results

    @call_immediately
    def test_export_measurement_data():
        # Test exporting measurements
        picarro.app.export_measurements(app_config)
        paths = list(app_config.output.get_path(OutItem.measurements_dir).iterdir())
        dataset_lengths = {
            len(pd.read_csv(path, index_col="datetime_utc"))  # type: ignore
            for path in paths
        }
        expected_lengths = {summary["n_samples"] for summary in expected_summaries}
        assert dataset_lengths == expected_lengths
