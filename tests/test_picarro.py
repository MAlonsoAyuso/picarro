import pytest
import picarro
import pathlib

_DATA_DIR = pathlib.Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


def test_read_raw():
    picarro.read_raw(data_path("example.dat"))


def test_require_unique_timestamps():
    with pytest.raises(ValueError):
        picarro.read_raw(data_path("duplicate_timestamp.dat"))


def test_split():
    d = picarro.read_raw(data_path("example.dat"))
    for part in picarro.split(d):
        assert len(part["solenoid_valves"].unique()) == 1
