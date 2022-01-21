import os
from pathlib import Path
import shutil
import pytest
import picarro.app
from picarro.config import AppConfig, UserConfig, ReadConfig, FitConfig, OutputConfig
from picarro.read import MeasurementMeta, PicarroColumns


config_example_src = Path(__file__).absolute().parent / "config_example.toml"
assert config_example_src.exists()

_EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"


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
                max_gap=5,
                min_length=1080,
                max_length=None,
            ),
            FitConfig(
                method="linear",
                t0=480,
                t0_margin=120,
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


def test_integrated(app_config: AppConfig, tmp_path: Path):
    data_dir = tmp_path / "data-dir" / "nested" / "path"
    shutil.copytree(_EXAMPLE_DATA_DIR / "adjacent_files", data_dir)

    # These were established by manually sifting through the files
    expected_summaries = [
        dict(solenoid_valve=13, length=217),
        dict(solenoid_valve=14, length=1789),
        dict(solenoid_valve=15, length=1787),
        dict(solenoid_valve=1, length=1779),
        dict(solenoid_valve=2, length=1782),
        dict(solenoid_valve=3, length=1789),
        dict(solenoid_valve=4, length=1786),
        dict(solenoid_valve=5, length=1783),
        dict(solenoid_valve=6, length=716),
    ]

    def summarize_measurement(mm: MeasurementMeta):
        return dict(solenoid_valve=mm.solenoid_valve, length=mm.length)

    measurement_summaries = [
        summarize_measurement(mm)
        for mm in picarro.app.iter_measurements_meta(app_config)
    ]

    assert measurement_summaries == expected_summaries

    data_summaries = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            length=len(m),
        )
        for m in picarro.app.iter_measurements(app_config)
    ]

    assert data_summaries == expected_summaries
