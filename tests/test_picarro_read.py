import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from picarro.read import (
    read_raw,
    iter_chunks,
    get_chunks_metadata,
    read_chunks_metadata,
    write_chunks_metadata,
    PicarroColumns,
    build_chunk_map,
    iter_measurements,
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


def test_iter_measurements():
    paths = (p for p in (_DATA_DIR / "adjacent_files").iterdir())
    chunk_map = build_chunk_map(paths)
    measurements = iter_measurements(chunk_map)  # using default max_gap

    # These were established by manually sifting through the files
    expected_chunks = [
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

    seen_chunks = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            length=len(m),
        )
        for m in measurements
    ]

    assert seen_chunks == expected_chunks


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = (p for p in (_DATA_DIR / "adjacent_files").iterdir())
    chunk_map = build_chunk_map(paths)

    # Timedelta 0 seconds ensures that chunks are too far apart and will be counted
    # as beloning to separate measurements
    measurements = iter_measurements(chunk_map, pd.Timedelta(0, "s"))

    # These were established by manually sifting through the files
    expected_chunks = [
        dict(solenoid_valve=13, length=217),
        dict(solenoid_valve=14, length=1789),
        dict(solenoid_valve=15, length=1787),
        dict(solenoid_valve=1, length=680),
        dict(solenoid_valve=1, length=1099),
        dict(solenoid_valve=2, length=1782),
        dict(solenoid_valve=3, length=1600),
        dict(solenoid_valve=3, length=189),
        dict(solenoid_valve=4, length=1786),
        dict(solenoid_valve=5, length=1783),
        dict(solenoid_valve=6, length=716),
    ]

    seen_chunks = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            length=len(m),
        )
        for m in measurements
    ]

    assert seen_chunks == expected_chunks
