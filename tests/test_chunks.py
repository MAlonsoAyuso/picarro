from __future__ import annotations
import itertools
import pytest
from picarro.config import MeasurementsConfig
from picarro.chunks import (
    ChunkMeta,
    read_chunks,
    _read_file,
    PicarroColumns,
)
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


CONFIG = MeasurementsConfig(
    valve_column=PicarroColumns.solenoid_valves,
    columns=[
        PicarroColumns.solenoid_valves,
        PicarroColumns.EPOCH_TIME,
        PicarroColumns.N2O,
    ]
)


def test_read_raw():
    _read_file(data_path("example.dat"), CONFIG)


def test_require_unique_timestamps():
    with pytest.raises(ValueError):
        _read_file(data_path("duplicate_timestamp.dat"), CONFIG)


def test_chunks_have_unique_int_valve_numbers():
    for chunk_meta, chunk in read_chunks(data_path("example.dat"), CONFIG).items():
        valve_numbers = chunk[CONFIG.valve_column]
        assert valve_numbers.dtype == int  # type: ignore
        assert len(valve_numbers.unique()) == 1


def test_chunk_metadata_is_correct():
    path = data_path("example.dat")

    # these have been manually verified to be the desired outcome
    expected_chunk_metas = [
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:01:15.170"),
            end=pd.Timestamp("2021-05-07 00:02:19.338000"),
            valve_number=5,
            n_samples=81,
        ),
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:02:21.696"),
            end=pd.Timestamp("2021-05-07 00:22:19.405000"),
            valve_number=6,
            n_samples=1487,
        ),
        ChunkMeta(
            path=path,
            start=pd.Timestamp("2021-05-07 00:22:20.719"),
            end=pd.Timestamp("2021-05-07 00:24:23.092000"),
            valve_number=7,
            n_samples=152,
        ),
    ]

    chunk_metas = list(read_chunks(path, CONFIG))
    assert expected_chunk_metas == list(chunk_metas)
