import pytest
from pandas.testing import assert_frame_equal
from picarro.read import (
    read_raw,
    iter_chunks,
    get_chunks_metadata,
    read_chunks_metadata,
    write_chunks_metadata,
    PicarroColumns,
)
import pathlib

_DATA_DIR = pathlib.Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


def test_read_raw():
    read_raw(data_path("example.dat"))


def test_require_unique_timestamps():
    with pytest.raises(ValueError):
        read_raw(data_path("duplicate_timestamp.dat"))


def test_chunks_have_unique_int_solenoid_valves():
    d = read_raw(data_path("example.dat"))
    for chunk in iter_chunks(d):
        solenoid_valves = chunk[PicarroColumns.solenoid_valves]
        assert solenoid_valves.dtype == int
        assert len(solenoid_valves.unique()) == 1


def test_chunk_metadata_is_correct():
    # example_chunks.csv is verified to be the desired outcome
    expected_chunks = read_chunks_metadata(data_path("example_chunks.csv"))
    d = read_raw(data_path("example.dat"))
    assert_frame_equal(get_chunks_metadata(d), expected_chunks)


def test_chunk_metadata_round_trip_file(tmp_path: pathlib.Path):
    file_path = tmp_path / "chunks.csv"
    d = read_raw(data_path("example.dat"))
    chunks_metadata = get_chunks_metadata(d)
    write_chunks_metadata(chunks_metadata, file_path)
    chunks_metadata_roundtripped = read_chunks_metadata(file_path)
    print(chunks_metadata.dtypes)
    print(chunks_metadata_roundtripped.dtypes)
    assert_frame_equal(chunks_metadata, chunks_metadata_roundtripped)
