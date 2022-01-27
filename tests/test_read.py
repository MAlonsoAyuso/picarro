from __future__ import annotations
import itertools
import pytest
from picarro.config import MeasurementsConfig
from picarro.chunks import (
    ChunkMeta,
    get_chunk_map,
    read_raw,
    PicarroColumns,
)
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


CONFIG = MeasurementsConfig(
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
    for chunk_meta, chunk in get_chunk_map(data_path("example.dat"), CONFIG).items():
        solenoid_valves = chunk[PicarroColumns.solenoid_valves]
        assert solenoid_valves.dtype == int  # type: ignore
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

    chunk_metas = list(get_chunk_map(path, CONFIG))
    assert expected_chunk_metas == list(chunk_metas)
