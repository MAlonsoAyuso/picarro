from __future__ import annotations
import itertools
import pytest
from picarro.config import ParsingConfig, ReadConfig
from picarro.read import (
    ChunkMeta,
    iter_measurement_metas,
    read_raw,
    iter_chunks,
    PicarroColumns,
    iter_measurements,
)
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


CONFIG = ReadConfig(
    columns=[
        PicarroColumns.solenoid_valves,
        PicarroColumns.EPOCH_TIME,
        PicarroColumns.N2O,
    ]
)


def test_read_raw():
    read_raw(data_path("example.dat"), CONFIG)


def test_require_unique_timestamps():
    with pytest.raises(ValueError):
        read_raw(data_path("duplicate_timestamp.dat"), CONFIG)


def test_chunks_have_unique_int_solenoid_valves():
    for chunk_meta, chunk in iter_chunks(data_path("example.dat"), CONFIG):
        solenoid_valves = chunk[PicarroColumns.solenoid_valves]
        assert solenoid_valves.dtype == int
        assert len(solenoid_valves.unique()) == 1


def test_chunk_metadata_is_correct():
    path = data_path("example.dat")

    # these have been manually verified to be the desired outcome
    expected_chunk_metas = [
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:01:15.170"),
            end=pd.Timestamp("2021-05-07 00:02:19.338000"),
            solenoid_valve=5,
            n_samples=81,
        ),
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:02:21.696"),
            end=pd.Timestamp("2021-05-07 00:22:19.405000"),
            solenoid_valve=6,
            n_samples=1487,
        ),
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:22:20.719"),
            end=pd.Timestamp("2021-05-07 00:24:23.092000"),
            solenoid_valve=7,
            n_samples=152,
        ),
    ]

    chunk_metas, _ = zip(*iter_chunks(path, CONFIG))
    assert expected_chunk_metas == list(chunk_metas)


def _test_measurements_and_summaries_correct(
    paths: list[Path], max_gap_s: int, expected_summaries: list[dict]
):
    def iter_chunk_metas():
        for path in paths:
            for chunk_meta, _ in iter_chunks(path, CONFIG):
                yield chunk_meta

    chunk_metas = iter_chunk_metas()
    measurement_metas = list(
        iter_measurement_metas(chunk_metas, max_gap=pd.Timedelta(max_gap_s, "s"))
    )
    meta_summaries = [
        dict(
            solenoid_valve=mm.solenoid_valve,
            n_samples=mm.n_samples,
        )
        for mm in measurement_metas
    ]

    assert meta_summaries == expected_summaries

    data_summaries = [
        dict(
            solenoid_valve=m[PicarroColumns.solenoid_valves].unique()[0],
            n_samples=len(m),
        )
        for m in iter_measurements(measurement_metas, CONFIG)
    ]

    assert data_summaries == expected_summaries


def test_iter_measurement_metas():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    max_gap_s = 5

    # These were established by manually sifting through the files
    expected_summaries = [
        dict(solenoid_valve=13, n_samples=217),
        dict(solenoid_valve=14, n_samples=1789),
        dict(solenoid_valve=15, n_samples=1787),
        dict(solenoid_valve=1, n_samples=1779),
        dict(solenoid_valve=2, n_samples=1782),
        dict(solenoid_valve=3, n_samples=1789),
        dict(solenoid_valve=4, n_samples=1786),
        dict(solenoid_valve=5, n_samples=1783),
        dict(solenoid_valve=6, n_samples=716),
    ]

    _test_measurements_and_summaries_correct(paths, max_gap_s, expected_summaries)


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    max_gap_s = 1
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(solenoid_valve=13, n_samples=217),
        dict(solenoid_valve=14, n_samples=1789),
        dict(solenoid_valve=15, n_samples=1787),
        dict(solenoid_valve=1, n_samples=680),
        dict(solenoid_valve=1, n_samples=1099),
        dict(solenoid_valve=2, n_samples=1782),
        dict(solenoid_valve=3, n_samples=1600),
        dict(solenoid_valve=3, n_samples=189),
        dict(solenoid_valve=4, n_samples=1786),
        dict(solenoid_valve=5, n_samples=1783),
        dict(solenoid_valve=6, n_samples=716),
    ]

    _test_measurements_and_summaries_correct(paths, max_gap_s, expected_summaries)
