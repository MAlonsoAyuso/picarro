from __future__ import annotations
import itertools

import pytest
from picarro.config import MeasurementsConfig
from picarro.core import ConfigProblem
from picarro.measurements import (
    build_measurement_metas,
    read_measurements,
)
from picarro.chunks import read_chunks
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "example_data"


def data_path(relpath: str):
    return _DATA_DIR / relpath


EXTRA_COLUMNS = [
    "N2O",
]


def test_iter_measurement_metas():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        extra_columns=EXTRA_COLUMNS,
        max_gap=pd.Timedelta(5, "s"),
    )
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(valve_number=13, n_samples=217),
        dict(valve_number=14, n_samples=1789),
        dict(valve_number=15, n_samples=1787),
        dict(valve_number=1, n_samples=1776),  # chunks from two files joined
        dict(valve_number=2, n_samples=1782),
        dict(valve_number=3, n_samples=1785),  # chunks from two files joined
        dict(valve_number=4, n_samples=1786),
        dict(valve_number=5, n_samples=1783),
        dict(valve_number=6, n_samples=711),
    ]

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def test_dont_join_chunks_if_time_gap_is_too_large():
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        extra_columns=EXTRA_COLUMNS,
        max_gap=pd.Timedelta(3, "s"),
    )
    # These were established by manually sifting through the files
    expected_summaries = [
        dict(valve_number=13, n_samples=217),
        dict(valve_number=14, n_samples=1789),
        dict(valve_number=15, n_samples=1787),
        dict(valve_number=1, n_samples=680),  # this is a non-connection of two files
        dict(valve_number=1, n_samples=1096),
        dict(valve_number=2, n_samples=1782),
        dict(valve_number=3, n_samples=1596),  # this is a non-connection of two files
        dict(valve_number=3, n_samples=189),
        dict(valve_number=4, n_samples=1786),
        dict(valve_number=5, n_samples=1783),
        dict(valve_number=6, n_samples=672),  # this is a cut inside a file
        dict(valve_number=6, n_samples=39),
    ]

    _test_measurements_and_summaries_correct(paths, config, expected_summaries)


def _test_measurements_and_summaries_correct(
    paths: list[Path], config: MeasurementsConfig, expected_summaries: list[dict]
):
    chunk_metas = itertools.chain(*(read_chunks(path, config) for path in paths))
    measurement_metas = list(build_measurement_metas(chunk_metas, config))

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


def test_valve_labels():
    valve_labels = {
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        223: "an-extra-does-not-hurt",
    }
    paths = [p for p in (_DATA_DIR / "adjacent_files").iterdir()]
    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        extra_columns=EXTRA_COLUMNS,
        max_gap=pd.Timedelta(1, "s"),
        valve_labels=valve_labels,
    )
    chunk_metas = list(itertools.chain(*(read_chunks(path, config) for path in paths)))
    for mm in build_measurement_metas(chunk_metas, config):
        assert mm.valve_label == valve_labels[mm.valve_number]

    config = MeasurementsConfig(
        valve_column="solenoid_valves",
        extra_columns=EXTRA_COLUMNS,
        max_gap=pd.Timedelta(1, "s"),
        valve_labels={1: "others-missing"},
    )
    with pytest.raises(ConfigProblem):
        list(build_measurement_metas(chunk_metas, config))

    with pytest.raises(ConfigProblem):
        config = MeasurementsConfig(
            valve_column="solenoid_valves",
            extra_columns=EXTRA_COLUMNS,
            max_gap=pd.Timedelta(1, "s"),
            valve_labels={i: "//: <- forbidden characters" for i in range(16)},
        )
