from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import (
    Iterable,
    Iterator,
    List,
    NewType,
    cast,
    Union,
)
import logging
import pandas as pd
from picarro.config import InvalidRowHandling, ParsingConfig, PicarroColumns, ReadConfig

logger = logging.getLogger(__name__)

INDEX_NAME = "datetime_utc"

# DataFile: A DataFrame from a whole .dat file (after some basic parsing)
# Chunk: A DataFrame with a contiguous subset of a DataFile,
#   with exactly one solenoid valve value.
# Measurement: A DataFrame possibly constructed from one or more chunks,
#   hopefully corresponding to a whole measurement from start to end.
DataFile = NewType("DataFile", pd.DataFrame)
Chunk = NewType("Chunk", pd.DataFrame)
Measurement = NewType("Measurement", pd.DataFrame)

_DATETIME64_UNIT = "ms"


@dataclass(frozen=True)
class ChunkMeta:
    path: Path
    start: pd.Timestamp
    end: pd.Timestamp
    solenoid_valve: int
    n_samples: int


@dataclass
class MeasurementMeta:
    chunks: List[ChunkMeta]
    start: pd.Timestamp
    end: pd.Timestamp
    solenoid_valve: int
    n_samples: int

    @staticmethod
    def from_chunk_metas(chunk_metas: List[ChunkMeta]) -> MeasurementMeta:
        solenoid_valves = {c.solenoid_valve for c in chunk_metas}
        assert len(solenoid_valves) == 1, solenoid_valves
        (solenoid_valve,) = solenoid_valves
        return MeasurementMeta(
            chunk_metas,
            chunk_metas[0].start,
            chunk_metas[-1].end,
            solenoid_valve,
            sum(c.n_samples for c in chunk_metas),
        )


class CannotParse(ValueError):
    pass


class InvalidData(CannotParse):
    pass


_ALWAYS_READ_COLUMNS = [PicarroColumns.EPOCH_TIME, PicarroColumns.solenoid_valves]


def _get_columns_to_read(user_columns: List[str]) -> List[str]:
    extra = [c for c in _ALWAYS_READ_COLUMNS if c not in set(user_columns)]
    return user_columns + extra


def _clean_raw_data(d: pd.DataFrame, config: ParsingConfig) -> pd.DataFrame:
    file_line_numbers = pd.RangeIndex(2, len(d) + 2)  # for debugging
    d = d.set_index(file_line_numbers)

    # Extract requested columns
    columns_to_read = _get_columns_to_read(config.columns)
    missing_columns = set(config.columns) - set(columns_to_read)
    if missing_columns:
        raise InvalidData(f"Missing columns {missing_columns}.")
    d = d[columns_to_read]

    # Reindex as time stamp
    d = d.pipe(_reindex_timestamp)

    # Nulls
    row_has_null = d.isnull().any(axis=1)
    if row_has_null.any():
        if config.null_rows == InvalidRowHandling.error:
            row_num = row_has_null.loc[lambda x: x].index[0]
            raise InvalidData(f"Missing value(s) in row {row_num}. {d.loc[row_num]}")
        elif config.null_rows == InvalidRowHandling.skip:
            n_violators = row_has_null.sum()
            logger.warning(f"Skipping {n_violators} lines with null values.")
            d = d.loc[~row_has_null]

    return d


def read_raw(path: Union[PathLike, str], config: ParsingConfig) -> DataFile:
    logger.info(f"read_raw {path}")
    d = pd.read_csv(path, sep=r"\s+")
    try:
        d = _clean_raw_data(d, config)
    except Exception as e:
        raise CannotParse(f"{path}: {e}") from e
    return cast(DataFile, d)


def _reindex_timestamp(d):
    # Reindex data in timestamps (numpy.datetime64).
    # Just to make sure, we also check that the resulting index is unique.

    # The Picarro data is in seconds with three decimals.
    # In order to exactly represent this data as a timestamp, we do the
    # conversion by first converting to integer milliseconds.
    timestamp = pd.to_datetime(
        d[PicarroColumns.EPOCH_TIME]
        .mul(1e3)
        .round()
        .astype("int64")
        .rename(INDEX_NAME),
        unit=_DATETIME64_UNIT,
    )
    if not timestamp.is_unique:
        first_duplicate = timestamp.loc[timestamp.duplicated()].iloc[0]
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    return d.set_index(timestamp)


def iter_chunks(path: Path, config: ReadConfig) -> Iterator[tuple[ChunkMeta, Chunk]]:
    logger.info(f"iter_chunks {path}")
    d = read_raw(path, config)
    d = d.pipe(_drop_data_between_valves)
    valve_just_changed = d[PicarroColumns.solenoid_valves].diff() != 0
    valve_change_count = valve_just_changed.cumsum()
    for i, chunk in d.groupby(valve_change_count):  # type: ignore
        chunk_meta = _build_chunk_metadata(chunk, path)
        yield chunk_meta, chunk


def _drop_data_between_valves(data):
    # Column "solenoid_valves" is sometimes noninteger for a short time when switching
    # from one valve to the next. Let's drop these data as they cannot be connected
    # to a chamber.
    valve_num = data[PicarroColumns.solenoid_valves]
    is_between_valves = valve_num.astype(int) != valve_num
    return data[~is_between_valves].astype({PicarroColumns.solenoid_valves: int})


def _build_chunk_metadata(chunk: Chunk, path: Path):
    solenoid_valves = chunk[PicarroColumns.solenoid_valves].unique()
    assert len(solenoid_valves) == 1, solenoid_valves
    (the_valve,) = solenoid_valves
    return ChunkMeta(
        path,
        chunk.index[0],
        chunk.index[-1],
        int(the_valve),
        len(chunk),
    )


def iter_measurement_metas(
    chunks: Iterable[ChunkMeta], max_gap: pd.Timedelta
) -> Iterator[MeasurementMeta]:
    chunks = list(chunks)
    chunks.sort(key=lambda c: c.start.to_numpy())

    while chunks:
        collected = [chunks.pop(0)]

        while chunks:
            prev_chunk = collected[-1]
            candidate = chunks.pop(0)

            time_gap = candidate.start - prev_chunk.end  # type: ignore
            if time_gap < pd.Timedelta(0):
                raise ValueError(f"overlapping chunks: {prev_chunk} {candidate}")

            is_adjacent = time_gap < max_gap
            same_valve = prev_chunk.solenoid_valve == candidate.solenoid_valve

            if is_adjacent and same_valve:
                collected.append(candidate)
            else:
                chunks.insert(0, candidate)
                break

        yield MeasurementMeta.from_chunk_metas(collected)


def iter_measurements(
    measurement_metas: Iterable[MeasurementMeta],
    config: ReadConfig,
) -> Iterator[Measurement]:
    read_cache = {}

    def read_file_into_cache(path: Path):
        read_cache.update(iter_chunks(path, config))

    def read_chunk(chunk_meta: ChunkMeta):
        if chunk_meta not in read_cache:
            read_file_into_cache(chunk_meta.path)

        chunk = read_cache[chunk_meta]
        return chunk

    for measurement_meta in measurement_metas:
        measurement = pd.concat(list(map(read_chunk, measurement_meta.chunks)))
        yield cast(Measurement, measurement)
