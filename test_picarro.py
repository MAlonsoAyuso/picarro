import pytest
import picarro


def test_parse_file():
    picarro.parse("example_data/example.dat")


def test_require_well_formatted_timestamps():
    with pytest.raises(ValueError):
        picarro.parse("example_data/misformatted_timestamp.dat")


def test_require_unique_timestamps():
    with pytest.raises(ValueError):
        picarro.parse("example_data/duplicate_timestamp.dat")
