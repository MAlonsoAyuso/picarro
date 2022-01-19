import itertools
import pytest
from picarro.read import (
    get_chunks_metadata,
    iter_measurements_meta,
    load_measurements_meta,
    read_raw,
    iter_chunks,
    load_chunks_meta,
    save_chunks_meta,
    PicarroColumns,
    iter_measurements,
    save_measurements_meta,
)
import pathlib
import pandas as pd

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
    # example_chunks.json is verified to be the desired outcome
    path = data_path("example.dat")
    expected_chunks = load_chunks_meta(data_path("example_chunks.json"))
    data = read_raw(path)
    chunks = get_chunks_metadata(data, "example.dat")
    assert expected_chunks == chunks


def test_chunk_metadata_round_trip_file(tmp_path: pathlib.Path):
    file_path = tmp_path / "chunks.json"
    d = read_raw(data_path("example.dat"))
    chunks_metadata = get_chunks_metadata(d, "example.dat")
    save_chunks_meta(chunks_metadata, file_path)
    chunks_metadata_roundtripped = load_chunks_meta(file_path)
    assert chunks_metadata_roundtripped == chunks_metadata


def test_iter_measurements():
    paths = (p for p in (_DATA_DIR / "adjacent_files").iterdir())
    chunks_meta = itertools.chain(
        *(get_chunks_metadata(read_raw(path), path) for path in paths)
    )
    measurements_meta = iter_measurements_meta(
        chunks_meta, max_gap=pd.Timedelta(5, "s")
    )
    measurements = iter_measurements(measurements_meta)

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
    chunks_meta = itertools.chain(
        *(get_chunks_metadata(read_raw(path), path) for path in paths)
    )
    measurements_meta = iter_measurements_meta(chunks_meta, max_gap=pd.Timedelta(0))
    measurements = iter_measurements(measurements_meta)

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


def test_measurements_meta_round_trip(tmp_path: pathlib.Path):
    file_path = tmp_path / "measurements.json"
    d = read_raw(data_path("example.dat"))
    chunks_metadata = get_chunks_metadata(d, "example.dat")
    measurements_meta = list(
        iter_measurements_meta(chunks_metadata, max_gap=pd.Timedelta(0))
    )
    save_measurements_meta(measurements_meta, file_path)
    measurements_meta_roundtripped = load_measurements_meta(file_path)
    assert measurements_meta_roundtripped == measurements_meta
