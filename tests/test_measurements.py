from __future__ import annotations
import itertools
from picarro.measurements import (
    MeasurementsConfig,
    read_measurements,
    stitch_chunk_metas,
)
from picarro.chunks import PicarroColumns, get_chunk_map
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath):
    return _DATA_DIR / relpath


COLUMNS = [
    PicarroColumns.solenoid_valves,
    PicarroColumns.EPOCH_TIME,
    PicarroColumns.N2O,
]


def test_iter_measurement_metas():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(columns=COLUMNS, max_gap=pd.Timedelta(5, "s"))
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

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(columns=COLUMNS, max_gap=pd.Timedelta(1, "s"))
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

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def _test_measurements_and_summaries_correct(
    paths: list[Path], config: MeasurementsConfig, expected_summaries: list[dict]
):
    chunk_metas = itertools.chain(*(get_chunk_map(path, config) for path in paths))
    measurement_metas = list(stitch_chunk_metas(chunk_metas, config))

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
        for m in read_measurements(measurement_metas, config)
    ]
    assert data_summaries == expected_summaries
