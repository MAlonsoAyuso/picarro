from ast import Call
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
    ReadConfig,
    FluxEstimationConfig,
    OutputConfig,
)
from picarro.read import MeasurementMeta, PicarroColumns
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
    return AppConfig.from_toml(config_path)


def test_create_config(app_config: AppConfig, tmp_path: Path):
    expected_conf = AppConfig(
        src_dir=tmp_path,
        results_subdir="config-filename-stem",
        user=UserConfig(
            ReadConfig(
                src="data-dir/**/*.dat",
                columns=["N2O", "CH4"],
                max_gap=pd.Timedelta(5, "s"),
                min_length=pd.Timedelta(1080, "s"),
                max_length=None,
            ),
            FluxEstimationConfig(
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

    assert app_config.cache_dir_absolute.is_absolute(), app_config.cache_dir_absolute
    assert (
        app_config.results_dir_absolute.is_absolute()
    ), app_config.results_dir_absolute


def call_immediately(func: Callable[[], None]) -> None:
    func()


def summarize_measurement(mm: MeasurementMeta):
    return dict(solenoid_valve=mm.solenoid_valve, length=mm.length)


def test_integrated(app_config: AppConfig, tmp_path: Path):
    data_dir = tmp_path / "data-dir" / "nested" / "path"
    shutil.copytree(_EXAMPLE_DATA_DIR / "adjacent_files", data_dir)

    # These were established by manually sifting through the files
    expected_summaries = [
        # one removed here compared to the full set, because it's too short
        dict(solenoid_valve=14, length=1789),
        dict(solenoid_valve=15, length=1787),
        dict(solenoid_valve=1, length=1779),
        dict(solenoid_valve=2, length=1782),
        dict(solenoid_valve=3, length=1789),
        dict(solenoid_valve=4, length=1786),
        dict(solenoid_valve=5, length=1783),
        # one removed here compared to the full set, because it's too short
    ]

    @call_immediately
    def test_measurement_metadata():
        # Check that measurement metadata objects are as expected
        measurement_summaries = [
            summarize_measurement(mm)
            for mm in picarro.app.iter_measurements_meta(app_config)
        ]

        assert measurement_summaries == expected_summaries

    @call_immediately
    def test_measurement_data():
        # Check that measurement datasets are as expected
        data_summaries = [
            dict(
                solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
                length=len(m),
            )
            for m in picarro.app.iter_measurements(app_config)
        ]

        assert data_summaries == expected_summaries

    @call_immediately
    def test_analysis_working():
        # Test analysis
        analysis_results = list(picarro.app.iter_analysis_results(app_config))
        expected_analysis_results = list(
            itertools.product(expected_summaries, app_config.user.measurements.columns)
        )
        seen_analysis_results = [
            (summarize_measurement(ar.measurement_meta), ar.estimator.column)
            for ar in analysis_results
        ]
        assert expected_analysis_results == seen_analysis_results

    @call_immediately
    def test_will_not_overwrite():
        # Test that it won't write into a non-empty results directory
        app_config.results_dir_absolute.mkdir()
        blocking_file_path = app_config.results_dir_absolute / "file"
        blocking_file_path.touch()
        with pytest.raises(FileExistsError):
            picarro.app.export_measurements(app_config)
        os.remove(blocking_file_path)

    @call_immediately
    def test_export_measurement_data():
        # Test exporting measurements
        picarro.app.export_measurements(app_config)
        paths = list((app_config.results_dir_absolute / "measurements").iterdir())
        assert len(paths) == len(expected_summaries)
        for path, summary in zip(sorted(paths), expected_summaries):
            data = pd.read_csv(path, index_col="EPOCH_TIME")
            assert list(data.columns) == [
                *app_config.user.measurements.columns,
                *app_config.user.output.export_columns_extra,
            ]
            assert len(data) == summary["length"]
            (valve,) = data[PicarroColumns.solenoid_valves].unique()
            assert valve == summary["solenoid_valve"]
