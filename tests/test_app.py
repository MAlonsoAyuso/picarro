import os
from pathlib import Path
import shutil
import itertools
from typing import Callable
import pytest
import picarro.app
from picarro.config import (
    AppConfig,
    UserConfig,
    FluxEstimationConfig,
    OutputConfig,
)
from picarro.measurements import MeasurementMeta, MeasurementsConfig
import numpy as np
import pandas as pd

from picarro.chunks import PicarroColumns

config_example_src = Path(__file__).absolute().parent / "config_example.toml"
assert config_example_src.exists()

_EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"


def abs_rel_diff(a, b):
    return np.abs((a - b) / b)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config_path = tmp_path / "config-filename-stem.toml"
    shutil.copyfile(config_example_src, config_path)
    return AppConfig.from_toml(config_path)


def test_create_config(app_config: AppConfig, tmp_path: Path):
    expected_conf = AppConfig.create(
        base_dir=tmp_path,
        user_config=UserConfig(
            MeasurementsConfig(
                src="data-dir/**/*.dat",
                columns=["N2O", "CH4", "CO2", "EPOCH_TIME", "solenoid_valves"],
                max_gap=pd.Timedelta(5, "s"),
                min_duration=pd.Timedelta(1080, "s"),
                max_duration=None,
            ),
            FluxEstimationConfig(
                columns=["N2O", "CH4"],
                method="linear",
                t0_delay=pd.Timedelta(480, "s"),
                t0_margin=pd.Timedelta(120, "s"),
                A=0.25,
                Q=4.16e-6,
                V=50e-3,
            ),
            OutputConfig(),
        ),
    )

    assert app_config == expected_conf


def call_immediately(func: Callable[[], None]) -> None:
    func()


def summarize_measurement(mm: MeasurementMeta):
    return dict(solenoid_valve=mm.solenoid_valve, n_samples=mm.n_samples)


def test_integrated(app_config: AppConfig, tmp_path: Path):
    data_dir = tmp_path / "data-dir" / "nested" / "path"
    shutil.copytree(_EXAMPLE_DATA_DIR / "adjacent_files", data_dir)

    @call_immediately
    def test_will_not_overwrite():
        # Test that it won't do anything if there is a non-empty out directory
        # without a marker file
        out_dir = app_config.paths.out
        out_dir.mkdir()
        blocking_file_path = out_dir / "file"
        blocking_file_path.touch()
        with pytest.raises(FileExistsError):
            list(picarro.app.iter_measurements(app_config))
        os.remove(blocking_file_path)

    # These were established by manually sifting through the files
    expected_summaries = [
        # one removed here compared to the full set, because it's too short
        dict(solenoid_valve=14, n_samples=1789),
        dict(solenoid_valve=15, n_samples=1787),
        dict(solenoid_valve=1, n_samples=1779),
        dict(solenoid_valve=2, n_samples=1782),
        dict(solenoid_valve=3, n_samples=1789),
        dict(solenoid_valve=4, n_samples=1786),
        dict(solenoid_valve=5, n_samples=1783),
        # one removed here compared to the full set, because it's too short
    ]

    @call_immediately
    def test_measurement_metadata():
        # Check that measurement metadata objects are as expected
        measurement_summaries = [
            summarize_measurement(mm)
            for mm in picarro.app._iter_measurement_metas(app_config)
        ]

        assert measurement_summaries == expected_summaries

    @call_immediately
    def test_measurement_data():
        # Check that measurement datasets are as expected
        data_summaries = [
            dict(
                solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
                n_samples=len(m),
            )
            for m in picarro.app.iter_measurements(app_config)
        ]

        assert data_summaries == expected_summaries

    @call_immediately
    def test_analysis_working():
        # Test analysis
        analysis_results = list(picarro.app._iter_analysis_results(app_config))
        expected_analysis_results = list(
            itertools.product(
                expected_summaries, app_config.user.flux_estimation.columns
            )
        )
        seen_analysis_results = [
            (summarize_measurement(ar.measurement_meta), ar.estimator.column)
            for ar in analysis_results
        ]
        assert expected_analysis_results == seen_analysis_results

    @call_immediately
    def test_export_measurement_data():
        # Test exporting measurements
        picarro.app.export_measurements(app_config)
        paths = list(app_config.paths.out_measurements.iterdir())
        assert len(paths) == len(expected_summaries)
        for path, summary in zip(sorted(paths), expected_summaries):
            data = pd.read_csv(path, index_col="datetime_utc")
            assert list(data.columns) == app_config.user.measurements.columns
            assert len(data) == summary["n_samples"]
            assert str(data[PicarroColumns.solenoid_valves].dtype).startswith(
                "int"
            )  # pyright: reportGeneralTypeIssues=false
