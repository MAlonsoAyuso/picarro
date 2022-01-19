import itertools
import pytest
from picarro.read import (
    ChunkMeta,
    get_chunks_metadata,
    iter_measurements_meta,
    read_raw,
    iter_chunks,
    PicarroColumns,
    iter_measurements,
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
    path = data_path("example.dat")

    # these have been manually verified to be the desired outcome
    expected_chunks = [
        ChunkMeta(
            path="example.dat",
            start=pd.Timestamp("2021-05-07 00:01:15.170"),
            end=pd.Timestamp("2021-05-07 00:02:19.338000"),
            solenoid_valve=5,
        ),
        ChunkMeta(
            path="example.dat",
            start=pd.Timestamp("2021-05-07 00:02:21.696"),
            end=pd.Timestamp("2021-05-07 00:22:19.405000"),
            solenoid_valve=6,
        ),
        ChunkMeta(
            path="example.dat",
            start=pd.Timestamp("2021-05-07 00:22:20.719"),
            end=pd.Timestamp("2021-05-07 00:24:23.092000"),
            solenoid_valve=7,
        ),
    ]

    data = read_raw(path)
    chunks = get_chunks_metadata(data, "example.dat")
    assert expected_chunks == chunks


def test_iter_measurements():
    paths = (p for p in (_DATA_DIR / "adjacent_files").iterdir())
    chunks_meta = itertools.chain(
        *(get_chunks_metadata(read_raw(path), path) for path in paths)
    )
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

    measurements_meta = iter_measurements_meta(
        chunks_meta, max_gap=pd.Timedelta(5, "s")
    )
    measurements = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            length=len(m),
        )
        for m in iter_measurements(measurements_meta)
    ]

    assert measurements == expected_measurements


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
