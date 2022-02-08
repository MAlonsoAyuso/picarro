from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import datetime
import glob
from itertools import zip_longest
import os
from pathlib import Path
import sqlite3
import time
from typing import Iterator, Optional, Tuple, Union
import pandas as pd
import numpy as np
import importlib.resources
import picarro.resources
import picarro.sqliteutil
from picarro.core import DataProcessingProblem
import logging

from picarro.util import check_item_types

logger = logging.getLogger(__name__)

_PANDAS_MISSING_COLS_START = (
    "Usecols do not match columns, columns expected but not found: "
)
_PICARRO_EPOCH_TIME_COL = "EPOCH_TIME"
_DB_SAMPLE_INDEX_COLUMN = "sample_time"
_DB_SAMPLE_VALVE_NUMBER_COLUMN = "valve_number"
_DATETIME64_UNIT_MULTIPLIERS = {
    "ns": 1,
}
_NUMPY_DATETIME64_DTYPE = "datetime64[ns]"

Value = Union[str, int, float]


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
    path: Path, valve_column: str, columns: Union[list[str], tuple[str]]
) -> pd.DataFrame:
    columns_to_read = list({_PICARRO_EPOCH_TIME_COL, valve_column, *columns})
    data = read_picarro_file(path, columns_to_read)

    # Drop rows with null values
    rows_with_nulls = data.isnull().any(axis=1)
    null_count = rows_with_nulls.sum()
    if null_count > 1:
        logger.warning(f"Dropping {null_count} rows with nulls in '{path}'.")
    data = data.dropna()

    # Create time index
    data.set_index(
        data[_PICARRO_EPOCH_TIME_COL]
        .astype(
            "float64"
            # Ensure float64 type before the following calculations.
            # The Picarro epoch time data has millisecond precision.
            # float64 is more enough to exactly represent the integer
            # number of milliseconds about 285,000 years into the future. OK!
        )
        .mul(1e3)  # to milliseconds
        .round()
        .astype("int64")
        .mul(1_000_000)  # to nanoseconds
        .astype(_NUMPY_DATETIME64_DTYPE),
        verify_integrity=True,
        inplace=True,
    )
    data.rename_axis(index=_DB_SAMPLE_INDEX_COLUMN, inplace=True)

    # Drop any rows with non-integer valve numbers
    data = data.loc[data[valve_column].round() == data[valve_column]]
    data[_DB_SAMPLE_VALVE_NUMBER_COLUMN] = data[valve_column].astype(int)

    return data[[_DB_SAMPLE_VALVE_NUMBER_COLUMN, *columns]]


def setup_sqlite_adapters():
    sqlite3.register_adapter(pd.Timestamp, _datetime_to_epoch_ns)
    sqlite3.register_adapter(datetime.datetime, _datetime_to_epoch_ns)
    sqlite3.register_adapter(datetime.timedelta, _timedelta_to_ns)
    sqlite3.register_adapter(pd.Timedelta, _timedelta_to_ns)
    sqlite3.register_adapter(np.datetime64, _datetime64_to_epoch_ns)
    sqlite3.register_adapter(np.timedelta64, _timedelta64_to_ns)

    # The following is needed because numpy integers do not inherit int:
    sqlite3.register_adapter(np.int8, int)
    sqlite3.register_adapter(np.int16, int)
    sqlite3.register_adapter(np.int32, int)
    sqlite3.register_adapter(np.int64, int)


def create_or_open(sql_path: Path) -> sqlite3.Connection:
    setup_sqlite_adapters()
    existed = sql_path.exists()
    conn = sqlite3.connect(sql_path)
    if not existed:
        schema = importlib.resources.read_text(picarro.resources, "schema.sql")
        with conn:
            conn.executescript(schema)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def import_data(
    conn: sqlite3.Connection,
    path: Path,
    valve_column: str,
    columns: Union[list[str], tuple[str]],
):
    data = read_and_clean_picarro_file(path, valve_column, columns)
    assert data.index.name == _DB_SAMPLE_INDEX_COLUMN, data.index
    assert _DB_SAMPLE_VALVE_NUMBER_COLUMN in data.columns, data.columns

    # Ensure that the new data to be read does not overlap with existing data
    conn.set_trace_callback(print)
    print(data.index[0], repr(data.index[0]))
    (collisions,) = conn.execute(
        "select count(*) from unfiltered_sample where sample_time between ? and ?",
        (data.index[0], data.index[-1]),
    ).fetchone()
    if collisions:
        raise ValueError(
            f"Data between {data.index[0]} and {data.index[-1]} cannot "
            f"be imported as it overlaps {collisions} existing sample(s)."
        )
    print(collisions)

    picarro.sqliteutil.insert_dataframe(
        conn,
        "unfiltered_sample",
        data,
        ensure_columns=True,
    )


@dataclass
class FilterConfig:
    column: str
    min: Optional[Value] = None
    max: Optional[Value] = None


def apply_filter(conn: sqlite3.Connection, filter_config: FilterConfig):
    exclusion_criteria = []
    if filter_config.min is not None:
        exclusion_criteria.append(f"{filter_config.column} < :min")
    if filter_config.min is not None:
        exclusion_criteria.append(f":max < {filter_config.column}")

    filter_config_dict = dataclasses.asdict(filter_config)
    with conn:
        conn.execute(
            "delete from sample_exclusion where column = :column",
            filter_config_dict,
        )

        if not exclusion_criteria:
            return

        conn.execute(
            (
                "insert into sample_exclusion (sample_time, column) "
                f"select sample_time, :column as column "
                "from unfiltered_sample WHERE "
                f"{' OR '.join(f'({criterion})' for criterion in exclusion_criteria)}"
            ),
            dataclasses.asdict(filter_config),
        )


SegmentBounds = Tuple[int, int]


def identify_segments(conn: sqlite3.Connection, max_gap: datetime.timedelta):
    with conn:
        conn.execute("DELETE FROM segment")
        conn.executemany(
            "insert into segment (start, end) values (?, ?)",
            _generate_segment_bounds(conn, max_gap),
        )


def _generate_segment_bounds(
    conn: sqlite3.Connection, max_gap: datetime.timedelta
) -> Iterator[SegmentBounds]:
    chunks = _load_valve_chunks(conn)
    for chunk in chunks:
        segment_number = (
            chunk.diff()  # == time difference from previous value
            .gt(_timedelta_to_ns(max_gap))  # == True if time gap is larger than max_gap
            .cumsum()  # this becomes a numbering of the chunk segments
        )
        for _, segment in chunk.groupby(segment_number):
            start, end = segment.iloc[0], segment.iloc[-1]
            yield (start, end)


def _load_valve_chunks(conn: sqlite3.Connection) -> Iterator[pd.Series]:
    cur = conn.execute("select sample_time from valve_changes")
    (start_times,) = zip(*cur.fetchall())

    for start, next_start in zip_longest(start_times, start_times[1:]):
        end_criterion = "and sample_time < :next_start " if next_start else ""
        cur = conn.execute(
            f"select sample_time from filtered_sample "
            f"where sample_time >= :start "
            f"{end_criterion}"
            f"order by sample_time",
            dict(start=start, next_start=next_start),
        )
        (timestamps,) = zip(*cur.fetchall())
        yield pd.Series(timestamps)


def _iter_segment_data(
    conn: sqlite3.Connection,
    min_duration: Optional[datetime.timedelta] = None,
    max_duration: Optional[datetime.timedelta] = None,
) -> Iterator[pd.DataFrame]:
    duration_criteria = []
    if min_duration is not None:
        duration_criteria.append(":min_duration <= end_time - start_time")
    if max_duration is not None:
        duration_criteria.append("end_time - start_time <= :max_duration")

    cur = conn.execute(
        (
            "select start_time, end_time from segment"
            + (" where " if duration_criteria else "")
            + " and ".join(duration_criteria)
        ),
        dict(min_duration=min_duration, max_duration=max_duration),
    )

    segment_bounds = cur.fetchall()

    (columns,) = zip(
        *conn.execute(
            "select name from pragma_table_info('unfiltered_sample')"
        ).fetchall()
    )

    for start, end in segment_bounds:
        yield (
            picarro.sqliteutil.read_dataframe(
                conn,
                f"select {', '.join(columns)} from filtered_sample "
                "where :start <= sample_time and sample_time <= :end",
                dict(start=start, end=end),
            )
            .astype({_DB_SAMPLE_INDEX_COLUMN: _NUMPY_DATETIME64_DTYPE})
            .set_index(_DB_SAMPLE_INDEX_COLUMN)
        )


SQL_PATH = Path(__file__).parent.parent.parent / "test.sqlite"
src_path = Path(__file__).parent.parent.parent / "example_data" / "example.dat"

if __name__ == "__main__":
    src = "C:/Users/Rasmus/Documents/proj/agrogreen/picarro/campaign-analysis/Campaign Data/**/*202111*.dat"
    paths = glob.glob(src, recursive=True)
    total_file_size = sum(os.stat(path).st_size for path in paths)
    print(total_file_size / 1e6)

    timings = {}
    ongoing_timing = {}

    def add_timing(label: str):
        end_timing()
        ongoing_timing.update(dict(label=label, start=time.time()))

    def end_timing():
        now = time.time()
        if ongoing_timing:
            timings[ongoing_timing.pop("label")] = now - ongoing_timing.pop("start")

    conn = create_or_open(SQL_PATH)
    add_timing("import")
    # for path in paths:
    #     import_data(
    #         conn, Path(path), "solenoid_valves", ["N2O", "CH4_dry", "ALARM_STATUS"]
    #     )

    add_timing("filters")
    apply_filter(conn, FilterConfig("CH4_dry", min=2, max=4.5))
    apply_filter(conn, FilterConfig("ALARM_STATUS", max=0))

    add_timing("identify segments")
    identify_segments(conn, datetime.timedelta(seconds=2))

    add_timing("read segments")
    for segment in _iter_segment_data(
        conn,
        min_duration=datetime.timedelta(minutes=19.5),
        max_duration=datetime.timedelta(minutes=20.5),
    ):
        pass

    print(segment)
    end_timing()

    print(timings)
    # columns = ["solenoid_valves", "ALARM_STATUS", "N2O_dry", "CH4_dry", "CO2"]

    # valve_number = 0
    # with conn:
    #     for i in range(30 * 24 * 3600):
    #         if i % (20 * 60) == 0:
    #             valve_number += 1
    #             if valve_number == 20:
    #                 valve_number = 0
    #         t = datetime.datetime.fromtimestamp(i + 0.123456789)
    #         conn.execute(
    #             "insert into unfiltered_sample (epoch_ns, valve_number) "
    #             "values (?, ?)",
    #             (t, valve_number),
    #         )
    # conn.set_trace_callback(print)

    # max_gap = datetime.timedelta(seconds=1.8)
    # identify_segments(conn, max_gap=max_gap)

    # t0 = time.time()
    # vns = load_valve_numbers(conn)
    # t1 = time.time()

    # valve_changed = vns.diff() != 0
    # # large_gap = vns.index.to_series() > 1
    # large_gap = vns.index.to_series().diff() > pd.Timedelta(1, "s")
    # t2 = time.time()

    # segment_num = (valve_changed | large_gap).cumsum()
    # t3 = time.time()

    # segments = segment_num.groupby(segment_num).aggregate("min", "max")
    # t4 = time.time()

    # times = [t0, t1, t2, t3, t4]
    # for s, e in zip(times, times[1:]):
    #     print(e - s)

    # for path in glob.glob(src, recursive=True):
    #     path = Path(path)
    #     assert path.exists(), path
    #     print(path)
    #     with conn:
    #         import_data(conn, path, columns=columns)

    # unset_filter(conn, "CH4_dry")


def _datetime_to_epoch_ns(v: datetime.datetime) -> int:
    return int(np.datetime64(v, "ns").view("int64"))


def _numpy_time_to_ns(v: Union[np.datetime64, np.timedelta64]) -> int:
    unit, count = np.datetime_data(v)
    return int(v.view("int64")) * _DATETIME64_UNIT_MULTIPLIERS[unit] * count


def _datetime64_to_epoch_ns(v: np.datetime64) -> int:
    return _numpy_time_to_ns(v)


def _timedelta_to_ns(v: datetime.timedelta) -> int:
    return int(np.timedelta64(v, "ns").view("int64"))


def _timedelta64_to_ns(v: np.timedelta64) -> int:
    return _numpy_time_to_ns(v)
