from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    Iterator,
    List,
    Mapping,
    NewType,
    Union,
)
import logging
from dataclasses import field
import pandas as pd

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
    columns: List[str] = field(default_factory=list)
    null_rows: InvalidRowHandling = InvalidRowHandling.skip
    epoch_time_column: str = "EPOCH_TIME"


INDEX_NAME = "datetime_utc"

Column = str


class PicarroColumns:
    DATE = "DATE"
    TIME = "TIME"
    FRAC_DAYS_SINCE_JAN1 = "FRAC_DAYS_SINCE_JAN1"
    FRAC_HRS_SINCE_JAN1 = "FRAC_HRS_SINCE_JAN1"
    JULIAN_DAYS = "JULIAN_DAYS"
    EPOCH_TIME = "EPOCH_TIME"
    ALARM_STATUS = "ALARM_STATUS"
    INST_STATUS = "INST_STATUS"
    CavityPressure = "CavityPressure"
    CavityTemp = "CavityTemp"
    DasTemp = "DasTemp"
    EtalonTemp = "EtalonTemp"
    WarmBoxTemp = "WarmBoxTemp"
    species = "species"
    MPVPosition = "MPVPosition"
    OutletValve = "OutletValve"
    solenoid_valves = "solenoid_valves"
    N2O = "N2O"
    N2O_30s = "N2O_30s"
    N2O_1min = "N2O_1min"
    N2O_5min = "N2O_5min"
    N2O_dry = "N2O_dry"
    N2O_dry30s = "N2O_dry30s"
    N2O_dry1min = "N2O_dry1min"
    N2O_dry5min = "N2O_dry5min"
    CO2 = "CO2"
    CH4 = "CH4"
    CH4_dry = "CH4_dry"
    H2O = "H2O"
    NH3 = "NH3"
    ChemDetect = "ChemDetect"
    peak_1a = "peak_1a"
    peak_41 = "peak_41"
    peak_4 = "peak_4"
    peak15 = "peak15"
    ch4_splinemax = "ch4_splinemax"
    nh3_conc_ave = "nh3_conc_ave"


_DATETIME64_UNIT = "ms"


class CannotParse(ValueError):
    pass


class InvalidData(CannotParse):
    pass


# ParsedFile: A DataFrame from a whole .dat file (after some basic parsing)
ParsedFile = NewType("ParsedFile", pd.DataFrame)

# Chunk: A DataFrame with a contiguous subset of a DataFile,
#   with exactly one valve number.
Chunk = NewType("Chunk", pd.DataFrame)


def _read_file(path: Union[PathLike, str], config: ParsingConfig) -> ParsedFile:
    logger.debug(f"Reading file {path}")
    d = pd.read_csv(path, sep=r"\s+")
    try:
        d = _clean_raw_data(d, config)
    except Exception as e:
        raise CannotParse(f"{path}: {e}") from e
    return ParsedFile(d)


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
            row_num = row_has_null.loc[lambda x: x].index[
                0
            ]  # pyright: reportGeneralTypeIssues=false
            raise InvalidData(f"Missing value(s) in row {row_num}. {d.loc[row_num]}")
        elif config.null_rows == InvalidRowHandling.skip:
            n_violators = row_has_null.sum()
            logger.warning(f"Skipping {n_violators} lines with null values.")
            d = d.loc[~row_has_null]

    return d


_ALWAYS_READ_COLUMNS = [PicarroColumns.EPOCH_TIME]


def _get_columns_to_read(user_columns: List[str]) -> List[str]:
    extra = [c for c in _ALWAYS_READ_COLUMNS if c not in set(user_columns)]
    return user_columns + extra


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
        first_duplicate = timestamp.loc[timestamp.duplicated()].iloc[
            0
        ]  # pyright: reportGeneralTypeIssues=false
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    return d.set_index(timestamp)


def _split_file(src_path: Path, config: ParsingConfig) -> Iterator[Chunk]:
    d = _read_file(src_path, config)
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
