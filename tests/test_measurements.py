from __future__ import annotations
import itertools
from picarro.config import MeasurementsConfig
from picarro.measurements import (
    read_measurements,
    stitch_chunk_metas,
)
from picarro.chunks import read_chunks
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath: str):
    return _DATA_DIR / relpath


COLUMNS = [
    "solenoid_valves",
    "EPOCH_TIME",
    "N2O",
]


def test_iter_measurement_metas():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        columns=COLUMNS,
        max_gap=pd.Timedelta(5, "s"),
    )
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(valve_number=13, n_samples=217),
        dict(valve_number=14, n_samples=1789),
        dict(valve_number=15, n_samples=1787),
        dict(valve_number=1, n_samples=1779),
        dict(valve_number=2, n_samples=1782),
        dict(valve_number=3, n_samples=1789),
        dict(valve_number=4, n_samples=1786),
        dict(valve_number=5, n_samples=1783),
        dict(valve_number=6, n_samples=716),
    ]

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        columns=COLUMNS,
        max_gap=pd.Timedelta(1, "s"),
    )
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(valve_number=13, n_samples=217),
        dict(valve_number=14, n_samples=1789),
        dict(valve_number=15, n_samples=1787),
        dict(valve_number=1, n_samples=680),
        dict(valve_number=1, n_samples=1099),
        dict(valve_number=2, n_samples=1782),
        dict(valve_number=3, n_samples=1600),
        dict(valve_number=3, n_samples=189),
        dict(valve_number=4, n_samples=1786),
        dict(valve_number=5, n_samples=1783),
        dict(valve_number=6, n_samples=716),
    ]

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def _test_measurements_and_summaries_correct(
    paths: list[Path], config: MeasurementsConfig, expected_summaries: list[dict]
):
    chunk_metas = itertools.chain(*(read_chunks(path, config) for path in paths))
    measurement_metas = list(stitch_chunk_metas(chunk_metas, config))

    meta_summaries = [
        dict(
            valve_number=mm.valve_number,
            n_samples=mm.n_samples,
        )
        for mm in measurement_metas
    ]
    assert meta_summaries == expected_summaries

    data_summaries = [
        dict(
            valve_number=m[config.valve_column].unique()[0],
            n_samples=len(m),
        )
        for m in read_measurements(measurement_metas, config)
    ]
    assert data_summaries == expected_summaries
