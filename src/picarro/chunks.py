from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    Hashable,
    Iterator,
    List,
    Mapping,
    NewType,
    Sequence,
    Union,
)
import logging
from dataclasses import field
import pandas as pd

from picarro.core import DataProcessingProblem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkMeta:
    path: Path
    start: pd.Timestamp
    end: pd.Timestamp
    valve_number: int
    n_samples: int


def read_chunks(src_path: Path, config: ParsingConfig) -> Mapping[ChunkMeta, Chunk]:
    chunks = list(_split_file(src_path, config))
    chunk_metas = [_build_chunk_meta(chunk, src_path, config) for chunk in chunks]
    chunk_map = dict(zip(chunk_metas, chunks))
    logger.debug(f"Read {len(chunk_map)} chunks from {src_path}")
    return chunk_map


class InvalidRowHandling(Enum):
    skip = "skip"
    error = "error"


@dataclass
class ParsingConfig:
    valve_column: str
    extra_columns: List[str] = field(default_factory=list)
    null_rows: InvalidRowHandling = InvalidRowHandling.skip
    epoch_time_column: str = "EPOCH_TIME"


INDEX_NAME = "datetime_utc"

Column = str


_DATETIME64_UNIT = "ms"


# ParsedFile: A DataFrame from a whole .dat file (after some basic parsing)
ParsedFile = NewType("ParsedFile", pd.DataFrame)

# Chunk: A DataFrame with a contiguous subset of a DataFile,
#   with exactly one valve number.
Chunk = NewType("Chunk", pd.DataFrame)

_PANDAS_MISSING_COLS_START = (
    "Usecols do not match columns, columns expected but not found: "
)


def read_file(path: Union[PathLike, str], config: ParsingConfig) -> ParsedFile:
    logger.debug(f"Reading file {path!r}.")
    try:
        d = pd.read_csv(path, sep=r"\s+", usecols=_get_columns_to_read(config))
    except ValueError as e:
        msg = str(e)
        if msg.startswith(_PANDAS_MISSING_COLS_START):
            columns_str = msg.replace(_PANDAS_MISSING_COLS_START, "")
            raise DataProcessingProblem(
                f"Columns {columns_str} not found in '{path}'."
            ) from e
        else:
            raise
    try:
        d = _clean_raw_data(d, config, path)
    except Exception as e:
        logger.exception(f"Unhandled problem processing {path}: {e}.")
        raise DataProcessingProblem(f"Unhandled problem processing {path}: {e}.") from e
    return ParsedFile(d)


def _get_columns_to_read(config: ParsingConfig) -> List[str]:
    return _deduplicate_sequence(
        [config.epoch_time_column, config.valve_column, *config.extra_columns]
    )


def _clean_raw_data(d: pd.DataFrame, config: ParsingConfig, path: Path) -> pd.DataFrame:
    file_line_numbers = pd.RangeIndex(2, len(d) + 2)  # for debugging
    d = d.set_index(file_line_numbers)

    # Reindex as time stamp
    d = _reindex_timestamp(d, config)

    # Nulls
    row_has_null = d.isnull().any(axis=1)
    if row_has_null.any():
        if config.null_rows == InvalidRowHandling.error:
            row_num = row_has_null.loc[lambda x: x].index[0]
            raise DataProcessingProblem(
                f"Missing value(s) in row {row_num} in '{path}'."
            )
        elif config.null_rows == InvalidRowHandling.skip:
            n_violators = row_has_null.sum()
            logger.debug(
                f"Skipping {n_violators} line(s) with null values in {path!r}."
            )
            if n_violators > 1:
                logger.warning(
                    f"Skipping {n_violators} lines with null values in {path!r}."
                )
            d = d.loc[~row_has_null]

    return d


def _deduplicate_sequence(l: Sequence[Hashable]) -> List[Hashable]:
    seen = set()
    result = []
    for item in l:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _reindex_timestamp(d, config: ParsingConfig):
    # Reindex data in timestamps (numpy.datetime64).
    # Just to make sure, we also check that the resulting index is unique.

    # The Picarro data is in seconds with three decimals.
    # In order to exactly represent this data as a timestamp, we do the
    # conversion by first converting to integer milliseconds.
    timestamp = pd.to_datetime(
        d[config.epoch_time_column].mul(1e3).round().astype("int64").rename(INDEX_NAME),
        unit=_DATETIME64_UNIT,
    )
    if not timestamp.is_unique:
        first_duplicate = timestamp.loc[timestamp.duplicated()].iloc[
            0
        ]  # pyright: reportGeneralTypeIssues=false
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    return d.set_index(timestamp)


def _split_file(src_path: Path, config: ParsingConfig) -> Iterator[Chunk]:
    d = read_file(src_path, config)
    d = d.pipe(_drop_data_between_valves, config=config)
    valve_just_changed = d[config.valve_column].diff() != 0
    valve_change_count = valve_just_changed.cumsum()
    for i, chunk in d.groupby(valve_change_count):  # type: ignore
        yield chunk


def _drop_data_between_valves(data: ParsedFile, config: ParsingConfig):
    # At least "solenoid_valves" is sometimes noninteger for a short time when switching
    # from one valve to the next. Let's drop these data as they cannot be connected
    # to a chamber.
    valve_num = data[config.valve_column]
    is_between_valves = valve_num.astype(int) != valve_num
    return data[~is_between_valves].astype({config.valve_column: int})


def _build_chunk_meta(chunk: Chunk, path: Path, config: ParsingConfig) -> ChunkMeta:
    valve_numbers = chunk[config.valve_column].unique()
    assert len(valve_numbers) == 1, valve_numbers
    (the_valve,) = valve_numbers
    return ChunkMeta(
        path,
        chunk.index[0],
        chunk.index[-1],
        int(the_valve),
        len(chunk),
    )
