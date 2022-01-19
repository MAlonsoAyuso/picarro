from operator import le
from pathlib import Path
import pytest
import picarro.app
from picarro.config import AppConfig
from picarro.read import Measurement, PicarroColumns

config_example_src = Path(__file__).absolute().parent / "config_example.toml"


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_toml(config_example_src)


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

