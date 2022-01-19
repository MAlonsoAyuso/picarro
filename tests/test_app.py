from pathlib import Path
import shutil
import pytest
import picarro.app
from picarro.config import AppConfig, UserConfig, ReadConfig, FitConfig, OutputConfig
from picarro.read import Measurement, PicarroColumns

config_example_src = Path(__file__).absolute().parent / "config_example.toml"
assert config_example_src.exists()


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_toml(config_example_src)


def test_create_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    shutil.copyfile(config_example_src, config_path)

    conf = AppConfig.from_toml(config_path)
    expected_conf = AppConfig(
        src_dir=config_path.parent,
        results_subdir=config_path.stem,
        user=UserConfig(
            ReadConfig(
                src="../example_data/adjacent_files/**/*.dat",
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

    assert conf == expected_conf

    assert conf.cache_dir_absolute.is_absolute(), conf.cache_dir_absolute
    assert conf.results_dir_absolute.is_absolute(), conf.results_dir_absolute


def test_iter_measurements(app_config: AppConfig):
    # These were established by manually sifting through the files
    expected_measurements = [
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

    def summarize_measurement(m: Measurement):
        (solenoid_valve,) = m[PicarroColumns.solenoid_valves].unique()
        return dict(solenoid_valve=solenoid_valve, length=len(m))

    seen_measurements = [
        summarize_measurement(m) for m in picarro.app.iter_measurements(app_config)
    ]

    assert seen_measurements == expected_measurements
