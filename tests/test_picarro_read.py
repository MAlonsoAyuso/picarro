from __future__ import annotations
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
            length=81,
        ),
        ChunkMeta(
            path="example.dat",
            start=pd.Timestamp("2021-05-07 00:02:21.696"),
            end=pd.Timestamp("2021-05-07 00:22:19.405000"),
            solenoid_valve=6,
            length=1487,
        ),
        ChunkMeta(
            path="example.dat",
            start=pd.Timestamp("2021-05-07 00:22:20.719"),
            end=pd.Timestamp("2021-05-07 00:24:23.092000"),
            solenoid_valve=7,
            length=152,
        ),
    ]

    data = read_raw(path)
    chunks = get_chunks_metadata(data, "example.dat")
    assert expected_chunks == chunks


def _test_measurements_and_summaries_correct(
    paths: list[pathlib.Path], max_gap_s: int, expected_summaries: list[dict]
):
    chunks_meta = itertools.chain(
        *(get_chunks_metadata(read_raw(path), path) for path in paths)
    )
    measurements_meta = list(
        iter_measurements_meta(chunks_meta, max_gap=pd.Timedelta(max_gap_s, "s"))
    )
    meta_summaries = [
        dict(
            solenoid_valve=mm.solenoid_valve,
            length=mm.length,
        )
        for mm in measurements_meta
    ]

    assert meta_summaries == expected_summaries

    data_summaries = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            length=len(m),
        )
        for m in iter_measurements(measurements_meta)
    ]

    assert data_summaries == expected_summaries


def test_iter_measurement_metas():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    max_gap_s = 5

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

    _test_measurements_and_summaries_correct(paths, max_gap_s, expected_summaries)


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    max_gap_s = 1
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(solenoid_valve=13, length=217),
        dict(solenoid_valve=14, length=1789),
        dict(solenoid_valve=15, length=1787),
        dict(solenoid_valve=1, length=680),
        dict(solenoid_valve=1, length=1099),
        dict(solenoid_valve=2, length=1782),
        dict(solenoid_valve=3, length=1789),
        dict(solenoid_valve=4, length=1786),
        dict(solenoid_valve=5, length=1783),
        dict(solenoid_valve=6, length=716),
    ]

    _test_measurements_and_summaries_correct(paths, max_gap_s, expected_summaries)
