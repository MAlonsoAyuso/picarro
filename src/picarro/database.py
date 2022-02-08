from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import datetime
from itertools import zip_longest
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Optional, Tuple, Union
import pandas as pd
import numpy as np
import importlib.resources
import picarro.fluxes
import picarro.resources
import picarro.sqliteutil
from picarro.core import DataProcessingProblem
import logging

from picarro.util import check_item_types

logger = logging.getLogger(__name__)

EPOCH_TIME_COLUMN = "EPOCH_TIME"
EPOCH_TIME_UNIT = "s"
_PANDAS_MISSING_COLS_START = (
    "Usecols do not match columns, columns expected but not found: "
)
_DB_SAMPLE_INDEX_COLUMN = "sample_time"
_DB_SAMPLE_VALVE_NUMBER_COLUMN = "valve_number"

Value = Union[str, int, float]


@dataclass
class FilterParams:
    min: Optional[Value] = None
    max: Optional[Value] = None


@dataclass
class SegmentInfo:
    start: datetime.datetime
    end: datetime.datetime
    valve_number: int

    @property
    def duration(self) -> datetime.timedelta:
        return self.end - self.start


@dataclass
class SegmentingParams:
    max_gap: datetime.timedelta = datetime.timedelta(seconds=10)
    min_duration: Optional[datetime.timedelta] = datetime.timedelta(seconds=60)
    max_duration: Optional[datetime.timedelta] = None


def read_picarro_file(path: Path, columns: Optional[list[str]]) -> pd.DataFrame:
    if columns is not None:
        check_item_types(columns, str)
    logger.debug(f"Reading file {path!r}.")
    try:
        data = pd.read_csv(path, sep=r"\s+", usecols=columns)
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
    return data


def read_and_clean_picarro_file(
    path: Path,
    valve_column: str,
    extra_columns: Union[list[str], tuple[str]],
) -> pd.DataFrame:
    columns_to_read = list({EPOCH_TIME_COLUMN, valve_column, *extra_columns})
    data = read_picarro_file(path, columns_to_read)

    # Drop rows with null values
    rows_with_nulls = data.isnull().any(axis=1)
    null_count = rows_with_nulls.sum()
    if null_count > 1:
        logger.warning(f"Dropping {null_count} rows with nulls in '{path}'.")
    data = data.dropna()

    # Create time index
    data.set_index(
        data[EPOCH_TIME_COLUMN]  # assuming this is in seconds
        .mul(1e3)  # to milliseconds
        .round()
        .astype("int64")
        .view(f"datetime64[ms]"),
        verify_integrity=True,
        inplace=True,
    )
    data.rename_axis(index=_DB_SAMPLE_INDEX_COLUMN, inplace=True)

    # Drop any rows with non-integer valve numbers
    data = data.loc[data[valve_column].round() == data[valve_column]]
    data[_DB_SAMPLE_VALVE_NUMBER_COLUMN] = data[valve_column].astype(int)

    return data[[_DB_SAMPLE_VALVE_NUMBER_COLUMN, *extra_columns]]


# def datetime_to_epoch_ms(datetime_obj: Union[datetime.datetime, np.datetime64]) -> int:
#     return int(np.datetime64(datetime_obj, "ms").view("int64"))


# def epoch_ms_to_datetime(epoch_ms: int) -> datetime.datetime:
#     return datetime.datetime.fromtimestamp(epoch_ms / 1e3)


# def timedelta_to_ms(timedelta_obj: datetime.timedelta) -> int:
#     return round(timedelta_obj.total_seconds() * 1e3)


# def ms_to_timedelta(timedelta_ms: int) -> datetime.timedelta:
#     return datetime.timedelta(seconds=timedelta_ms / 1e3)


def configure_sqlite():
    # sqlite3.register_adapter(pd.Timestamp, datetime_to_epoch_ms)
    # sqlite3.register_adapter(datetime.datetime, datetime_to_epoch_ms)
    # sqlite3.register_adapter(datetime.timedelta, timedelta_to_ms)
    # sqlite3.register_adapter(np.datetime64, datetime_to_epoch_ms)

    sqlite3.register_adapter(pd.Timestamp, pd.Timestamp.isoformat)
    sqlite3.register_adapter(datetime.datetime, datetime.datetime.isoformat)
    sqlite3.register_adapter(datetime.timedelta, datetime.timedelta.total_seconds)
    sqlite3.register_adapter(np.datetime64, str)

    # The following is needed because numpy integers do not inherit int:
    sqlite3.register_adapter(np.int8, int)
    sqlite3.register_adapter(np.int16, int)
    sqlite3.register_adapter(np.int32, int)
    sqlite3.register_adapter(np.int64, int)


def create_or_open(sql_path: Path) -> sqlite3.Connection:
    configure_sqlite()
    existed = sql_path.exists()
    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    if not existed:
        schema = importlib.resources.read_text(picarro.resources, "schema.sql")
        with conn:
            conn.executescript(schema)
        conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def import_data(
    conn: sqlite3.Connection,
    path: Path,
    valve_column: str,
    extra_columns: Union[list[str], tuple[str]],
) -> int:
    data = read_and_clean_picarro_file(path, valve_column, extra_columns)
    assert data.index.name == _DB_SAMPLE_INDEX_COLUMN, data.index
    assert _DB_SAMPLE_VALVE_NUMBER_COLUMN in data.columns, data.columns

    # Ensure that the new data to be read does not overlap with existing data
    (collisions,) = conn.execute(
        "select count(*) from unfiltered_sample where sample_time between ? and ?",
        (data.index[0], data.index[-1]),
    ).fetchone()
    if collisions:
        raise ValueError(
            f"Data between {data.index[0]} and {data.index[-1]} cannot "
            f"be imported as it overlaps {collisions} existing sample(s)."
        )

    cur = picarro.sqliteutil.insert_dataframe(
        conn,
        "unfiltered_sample",
        data,
        ensure_columns=True,
    )

    return cur.rowcount


def remove_filters(conn: sqlite3.Connection):
    conn.execute("delete from sample_exclusion")


def apply_filter(conn: sqlite3.Connection, column: str, filter_params: FilterParams):
    exclusion_criteria = []
    if filter_params.min is not None:
        exclusion_criteria.append(f"{column} < :min")
    if filter_params.max is not None:
        exclusion_criteria.append(f":max < {column}")

    if not exclusion_criteria:
        return

    conn.execute(
        (
            "insert into sample_exclusion (sample_time, column) "
            f"select sample_time, :column as column "
            "from unfiltered_sample WHERE "
            f"{' OR '.join(f'({criterion})' for criterion in exclusion_criteria)}"
        ),
        dict(column=column, **dataclasses.asdict(filter_params)),
    )


def count_excluded_samples(conn: sqlite3.Connection, columns: list[str]) -> int:
    cur = conn.execute(
        "select count(*) from sample_exclusion "
        f"where column in ({', '.join('?'*len(columns))})",
        columns,
    )
    (n,) = cur.fetchone()
    return n


def count_included_samples(conn: sqlite3.Connection) -> int:
    cur = conn.execute("select count(*) from filtered_sample")
    (n,) = cur.fetchone()
    return n


def identify_segments(conn: sqlite3.Connection, segmenting_params: SegmentingParams):
    with conn:
        conn.execute("DELETE FROM segment")
        conn.executemany(
            "insert into segment (start_time, end_time, valve_number) "
            "values (:start, :end, :valve_number)",
            map(dataclasses.asdict, _generate_segment_infos(conn, segmenting_params)),
        )


def iter_segments_info(conn: sqlite3.Connection) -> Iterable[SegmentInfo]:
    cur = conn.execute("select start_time, end_time, valve_number from segment")
    for row in cur.fetchall():
        yield SegmentInfo(
            start=pd.Timestamp(row["start_time"]),
            end=pd.Timestamp(row["end_time"]),
            valve_number=row["valve_number"],
        )


def read_segment(conn: sqlite3.Connection, segment_info: SegmentInfo) -> pd.DataFrame:
    df = picarro.sqliteutil.read_dataframe(
        conn,
        "select * from filtered_sample where sample_time between :start and :end "
        "order by sample_time",
        dict(start=segment_info.start, end=segment_info.end),
    )
    valve_numbers = set(df[_DB_SAMPLE_VALVE_NUMBER_COLUMN].unique())
    assert valve_numbers == {segment_info.valve_number}, (valve_numbers, segment_info)
    return df.astype({_DB_SAMPLE_INDEX_COLUMN: "datetime64[ms]"}).set_index(
        _DB_SAMPLE_INDEX_COLUMN
    )


def save_flux_estimate(
    conn: sqlite3.Connection, flux_estimate: picarro.fluxes.FluxEstimator
):
    picarro.sqliteutil.insert_mapping(
        conn,
        "flux_estimate",
        dict(
            start_time=flux_estimate.moments.data_start,
            column=flux_estimate.column,
            params=json.dumps(flux_estimate.unstructure()),
        ),
        on_conflict="replace",
    )


def read_flux_estimate(
    conn: sqlite3.Connection, start_time: datetime.datetime, column: str
) -> picarro.fluxes.FluxEstimator:
    (params_json,) = conn.execute(
        "select params from flux_estimate "
        "where start_time = :start_time and column = :column",
        dict(start_time=start_time, column=column)
    ).fetchone()
    obj = json.loads(params_json)
    return picarro.fluxes.json_converter.structure(obj, picarro.fluxes.FluxEstimator)


def _generate_segment_infos(
    conn: sqlite3.Connection, params: SegmentingParams
) -> Iterator[SegmentInfo]:
    def duration_is_ok(segment_info: SegmentInfo):
        duration = segment_info.duration
        assert duration >= datetime.timedelta(0), duration
        if params.min_duration and duration < params.min_duration:
            return False
        if params.max_duration and duration > params.max_duration:
            return False
        return True

    for valve_number, chunk in _load_valve_chunks(conn):
        segment_number = (
            chunk.diff()  # == time difference from previous value
            .gt(params.max_gap)  # == True if time gap is larger than max_gap
            .cumsum()  # this becomes a numbering of the chunk segments
        )
        for _, segment in chunk.groupby(segment_number):
            start, end = segment.iloc[0], segment.iloc[-1]
            segment_info = SegmentInfo(start, end, valve_number)
            if duration_is_ok(segment_info):
                yield segment_info


def _load_valve_chunks(conn: sqlite3.Connection) -> Iterator[Tuple[int, pd.Series]]:
    cur = conn.execute("select sample_time, valve_number from valve_changes")
    valve_number_by_start_time = dict(cur.fetchall())
    start_times = list(valve_number_by_start_time)

    for start, next_start in zip_longest(start_times, start_times[1:]):
        assert next_start is None or start < next_start
        end_criterion = "and sample_time < :next_start " if next_start else ""
        cur = conn.execute(
            f"select sample_time from filtered_sample "
            f"where sample_time >= :start "
            f"{end_criterion}"
            f"order by sample_time",
            dict(start=start, next_start=next_start),
        )
        (timestamps,) = zip(*cur.fetchall())
        yield valve_number_by_start_time[start], pd.to_datetime(timestamps).to_series()
