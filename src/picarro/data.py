from __future__ import annotations

import datetime
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from picarro.util import format_duration

logger = logging.getLogger(__name__)

EPOCH_TIME_COLUMN = "EPOCH_TIME"
EPOCH_TIME_UNIT = "s"
_PANDAS_MISSING_COLS_START = (
    "Usecols do not match columns, columns expected but not found: "
)
TIMESTAMP_INDEX_NAME = "sample_time_utc"
VALVE_NUMBER_COLUMN = "valve_number"

Value = Union[int, float, datetime.datetime]

DEFAULT_MAX_GAP = datetime.timedelta(seconds=10)

Column = str
ValveNumber = int


class DataProcessingProblem(Exception):
    pass


@dataclass(frozen=True)
class FilterParams:
    min: Optional[Value] = None
    max: Optional[Value] = None
    allow: Optional[Tuple[Value, ...]] = None
    disallow: Tuple[Value, ...] = ()


@dataclass(frozen=True)
class FilterSummary:
    n_rows: int
    n_removed_by_col: dict[Column, int]
    n_removed_total: int


@dataclass(frozen=True)
class BlockInfo:
    path: Path
    start_time: datetime.datetime
    end_time: datetime.datetime
    valve_number: ValveNumber

    @property
    def duration(self) -> datetime.timedelta:
        return self.end_time - self.start_time

    @classmethod
    def from_block(cls, path: Path, df: pd.DataFrame):
        assert len(df) > 0, (path, df)
        valve_numbers = df[VALVE_NUMBER_COLUMN].unique()
        assert len(valve_numbers) == 1, valve_numbers
        (valve_number,) = valve_numbers
        return cls(
            path,
            df.index[0],
            df.index[-1],
            int(valve_number),
        )


@dataclass(frozen=True)
class MeasurementInfo:
    blocks: Tuple[BlockInfo, ...]

    def __post_init__(self):
        block_starts = tuple(b.start_time for b in self.blocks)
        if block_starts != tuple(sorted(block_starts)):
            raise ValueError(f"Blocks are not sorted: {self.blocks}")

        block_ends = tuple(b.end_time for b in self.blocks)
        gaps = [
            next_start - prev_end
            for prev_end, next_start in zip(block_ends[:-1], block_starts[1:])
        ]
        for gap in gaps:
            if gap <= datetime.timedelta(0):
                raise ValueError(f"Blocks overlap: {self.blocks}")

        valve_numbers = set(b.valve_number for b in self.blocks)
        assert len(valve_numbers) == 1, valve_numbers

    @property
    def data_start(self) -> datetime.datetime:
        return self.blocks[0].start_time

    @property
    def data_end(self) -> datetime.datetime:
        return self.blocks[-1].end_time

    @property
    def duration(self) -> datetime.timedelta:
        return self.data_end - self.data_start

    @property
    def valve_number(self) -> int:
        return self.blocks[0].valve_number


@dataclass(frozen=True)
class FilteredMeasurements:
    accepted: Tuple[MeasurementInfo]
    rejected: Tuple[MeasurementInfo]


# @dataclass
# class SegmentingParams:
#     max_gap: datetime.timedelta = datetime.timedelta(seconds=10)
#     min_duration: Optional[datetime.timedelta] = datetime.timedelta(seconds=60)
#     max_duration: Optional[datetime.timedelta] = None


def read_picarro_file(
    path: Path,
    valve_src_column: Column,
    extra_columns: Union[list[Column], tuple[Column]],
) -> pd.DataFrame:
    """
    Read a Picarro data file and do some basic parsing.

    Assumptions when parsing the file:
    - Fixed-width files with whitespace delimiters.
    - Lines should only have null values e.g. when file writing has ended abruptly;
      in other words, lines with null values should be discarded.
    - Lines are ordered by time, with rows uniquely identified by an epoch time stamp
      in seconds with three decimals (i.e., millisecond resolution).
    - There is a column indicating the current valve as a number,
      and all interesting data has an integer valve number. A (small) number of
      non-integer valve numbers may appear when the machine is switching valves, and
      these few lines can be discarded.

    Hence:
    - Read the data using whitespace delimiters.
    - Discard any lines with null values.
    - Transform the epoch time column to a datetime column with millisecond resolution.
    - Discard any lines with non-integer valve numbers.
    - Convert the valve number column to integer dtype.
    """
    columns_to_read = list({EPOCH_TIME_COLUMN, valve_src_column, *extra_columns})

    logger.debug(f"Reading file {path!r}.")
    try:
        data = pd.read_csv(path, sep=r"\s+", usecols=columns_to_read, engine="c")
    except ValueError as e:
        msg = str(e)
        if msg.startswith(_PANDAS_MISSING_COLS_START):
            columns_str = msg.replace(_PANDAS_MISSING_COLS_START, "")
            raise DataProcessingProblem(
                f"Columns {columns_str} not found in '{path}'."
            ) from e
        else:
            raise
    assert isinstance(data, pd.DataFrame)

    # Create timestamp index
    data[TIMESTAMP_INDEX_NAME] = (
        data[EPOCH_TIME_COLUMN]  # assuming this is in seconds
        .mul(1e3)  # to milliseconds
        .round()
        .astype("int64")
        .view(f"datetime64[ms]")
    )
    if not data[TIMESTAMP_INDEX_NAME].is_unique:
        duplicate_timestamps = data[TIMESTAMP_INDEX_NAME].loc[lambda d: d.duplicated()]
        raise ValueError(f"Duplicated timestamp in '{path}': {duplicate_timestamps}.")
    data = data.set_index(TIMESTAMP_INDEX_NAME)

    # Drop the EPOCH_TIME_COLUMN unless the user requested it to be read
    if EPOCH_TIME_COLUMN not in extra_columns:
        data = data.drop(columns=EPOCH_TIME_COLUMN)

    # Drop rows with null values
    row_has_null_value = data.isnull().any(axis=1)
    null_count = row_has_null_value.sum()
    if null_count > 1:
        logger.warning(f"Dropping {null_count} rows with nulls in '{path}'.")
    data = data[~row_has_null_value]

    # Drop any rows with non-integer valve numbers; then create VALVE_NUMBER_COLUMN
    if not str(data[valve_src_column].dtype).startswith("int"):
        data = data.loc[data[valve_src_column].round() == data[valve_src_column]]
        data[VALVE_NUMBER_COLUMN] = data[valve_src_column].astype(int)

    return data


def read_block(
    block_info: BlockInfo,
    valve_src_column: Column,
    extra_columns: Union[list[Column], tuple[Column]],
) -> pd.DataFrame:
    df = read_picarro_file(block_info.path, valve_src_column, extra_columns)
    return df.loc[block_info.start_time : block_info.end_time]


def read_measurement(
    measurement_info: MeasurementInfo,
    valve_src_column: Column,
    extra_columns: Union[list[Column], tuple[Column]],
) -> pd.DataFrame:
    blocks = (
        read_block(block_info, valve_src_column, extra_columns)
        for block_info in measurement_info.blocks
    )
    measurement = pd.concat(blocks)
    valve_numbers = measurement[VALVE_NUMBER_COLUMN].unique()
    assert len(valve_numbers) == 1, set(valve_numbers)
    return measurement


def _get_filter_removals_column(s: pd.Series, filter_params: FilterParams) -> pd.Series:
    logger.debug(f"Applying filter {filter_params} on {s.name}")
    excluded = pd.Series(data=False, index=s.index)
    if filter_params.allow:
        excluded |= ~s.isin(filter_params.allow)
    if filter_params.disallow:
        excluded |= s.isin(filter_params.disallow)
    if filter_params.min is not None:
        excluded |= s < filter_params.min  # pyright: reportGeneralTypeIssues=false
    if filter_params.max is not None:
        excluded |= filter_params.max < s
    return excluded


def get_data_filter_results(
    df: pd.DataFrame, filters: dict[Column, FilterParams]
) -> pd.DataFrame:
    removals_by_column = pd.DataFrame(
        {
            column: _get_filter_removals_column(df[column], filter_params)
            for column, filter_params in filters.items()
        }
    )
    return removals_by_column


def summarize_data_filter_results(filter_results: pd.DataFrame) -> FilterSummary:
    n_rows = len(filter_results)
    n_removed_total = int(filter_results.any(axis=1).sum())
    n_removed_by_col = filter_results.sum()
    return FilterSummary(n_rows, n_removed_by_col.to_dict(), n_removed_total)


def apply_filter_results(df: pd.DataFrame, filter_results: pd.DataFrame):
    return df.loc[~filter_results.any(axis=1)]


def split_data_to_blocks(
    df: pd.DataFrame, max_gap: datetime.timedelta
) -> list[pd.DataFrame]:
    if len(df) == 0:
        return []
    gap_is_large = df.index.to_series().diff() > max_gap
    valve_changed = df[VALVE_NUMBER_COLUMN].diff().fillna(0) != 0
    new_block_starts = gap_is_large | valve_changed
    block_numbers = new_block_starts.cumsum()
    _, blocks = zip(*df.groupby(block_numbers))
    return list(blocks)


def join_block_infos(
    block_infos: Iterable[BlockInfo],
    max_gap: datetime.timedelta,
) -> Iterator[MeasurementInfo]:

    block_infos = list(block_infos)
    if not len(block_infos):
        return

    block_infos.sort(key=lambda block_info: block_info.start_time)

    def blocks_should_be_joined(b1: BlockInfo, b2: BlockInfo) -> bool:
        if b1.valve_number != b2.valve_number:
            return False

        time_gap = b2.start_time - b1.end_time
        if time_gap < datetime.timedelta(0):
            raise DataProcessingProblem(f"Overlapping blocks: {b1} {b2}")

        return time_gap <= max_gap

    starts_new_measurement = np.zeros(len(block_infos), dtype=np.int32)
    starts_new_measurement[0] = 1

    for i in range(1, len(block_infos)):
        starts_new_measurement[i] = 1 - blocks_should_be_joined(
            block_infos[i - 1], block_infos[i]
        )

    measurement_numbers = starts_new_measurement.cumsum()
    measurement_number_by_block = dict(zip(block_infos, measurement_numbers))
    get_measurement_number = measurement_number_by_block.__getitem__

    for _, block_group in itertools.groupby(block_infos, get_measurement_number):
        yield MeasurementInfo(tuple(block_group))


def filter_measurements(
    measurement_infos: Iterable[MeasurementInfo],
    min_duration: Optional[datetime.timedelta],
    max_duration: Optional[datetime.timedelta],
) -> list[MeasurementInfo]:
    accepted = []
    for measurement_info in measurement_infos:
        duration = measurement_info.data_end - measurement_info.data_start
        if min_duration and duration < min_duration:
            continue
        if max_duration and max_duration < duration:
            continue
        logger.debug(
            f"Accepting measurement. "
            f"Start time: {measurement_info.data_start}. "
            f"Blocks: {len(measurement_info.blocks)}. "
            f"Duration: {format_duration(measurement_info.duration)}."
        )
        accepted.append(measurement_info)
    return accepted
