from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator, List, NewType, cast, Union
import pandas as pd
import numpy as np


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


CONC_UNITS = {
    PicarroColumns.CH4: "ppmv",
    PicarroColumns.CO2: "ppmv",
    PicarroColumns.N2O: "ppmv",
}

DEFAULT_MAX_GAP_BETWEEN_CHUNKS = pd.Timedelta(5, "s")

# DataFile: A DataFrame from a whole .dat file (after some basic parsing)
# Chunk: A DataFrame with a contiguous subset of a DataFile,
#   with exactly one solenoid valve value.
# Measurement: A DataFrame possibly constructed from one or more chunks,
#   hopefully corresponding to a whole measurement from start to end.
DataFile = NewType("DataFile", pd.DataFrame)
Chunk = NewType("Chunk", pd.DataFrame)
Measurement = NewType("Measurement", pd.DataFrame)

_DATETIME64_UNIT = "ms"


@dataclass
class ChunkMeta:
    path: str
    start: pd.Timestamp
    end: pd.Timestamp
    solenoid_valve: int

    def to_dict(self) -> dict[str, Union[str, int]]:
        obj = dataclasses.asdict(self)
        for key in ["start", "end"]:
            v = obj[key]
            assert isinstance(v, pd.Timestamp)
            obj[key] = str(v)
        return obj

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> ChunkMeta:
        for key in ["start", "end"]:
            v = pd.Timestamp(obj[key])
            obj[key] = v
        return ChunkMeta(**obj)


MeasurementMeta = List[ChunkMeta]


def read_raw(path: Union[PathLike, str]) -> DataFile:
    d = pd.read_csv(path, sep=r"\s+").pipe(_reindex_timestamp)
    return cast(DataFile, d)


def _reindex_timestamp(d):
    # Reindex data in timestamps (numpy.datetime64).
    # Just to make sure, we also check that the resulting index is unique.

    # The Picarro data is in seconds with three decimals.
    # In order to exactly represent this data as a timestamp, we do the
    # conversion by first converting to integer milliseconds.
    timestamp = pd.to_datetime(
        d[PicarroColumns.EPOCH_TIME].mul(1e3).round().astype("int64"),
        unit=_DATETIME64_UNIT,
    )
    if not timestamp.is_unique:
        first_duplicate = timestamp.loc[timestamp.duplicated()].iloc[0]
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    return d.set_index(timestamp)


def iter_chunks(d: DataFile) -> Iterator[Chunk]:
    d = d.pipe(_drop_data_between_valves)
    valve_just_changed = d[PicarroColumns.solenoid_valves].diff() != 0
    valve_change_count = valve_just_changed.cumsum()
    for i, g in d.groupby(valve_change_count):  # type: ignore
        yield g


def _drop_data_between_valves(data):
    # Column "solenoid_valves" is sometimes noninteger for a short time when switching
    # from one valve to the next. Let's drop these data as they cannot be connected
    # to a chamber.
    valve_num = data[PicarroColumns.solenoid_valves]
    is_between_valves = valve_num.astype(int) != valve_num
    return data[~is_between_valves].astype({PicarroColumns.solenoid_valves: int})


def _get_chunk_metadata(chunk: Chunk, path: str):
    solenoid_valves = chunk[PicarroColumns.solenoid_valves].unique()
    assert len(solenoid_valves) == 1, solenoid_valves
    (the_valve,) = solenoid_valves
    return ChunkMeta(
        path,
        chunk.index[0],
        chunk.index[-1],
        int(the_valve),
    )


def get_chunks_metadata(data: DataFile, path: Union[Path, str]) -> List[ChunkMeta]:
    return [_get_chunk_metadata(chunk, str(path)) for chunk in iter_chunks(data)]


def save_chunks_meta(chunks_meta: List[ChunkMeta], path: Path):
    with open(path, "x") as f:
        f.write(json.dumps([c.to_dict() for c in chunks_meta]))


def load_chunks_meta(path: Path) -> List[ChunkMeta]:
    with open(path, "r") as f:
        data = json.loads(f.read())
    return [ChunkMeta.from_dict(item) for item in data]


def iter_measurements_meta(
    chunks: Iterable[ChunkMeta], max_gap: pd.Timedelta = DEFAULT_MAX_GAP_BETWEEN_CHUNKS
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

        yield collected


def iter_measurements(
    measurement_metas: Iterable[MeasurementMeta],
) -> Iterator[Measurement]:
    read_cache = {
        "path": Path(),
        "data": pd.DataFrame(),
    }

    def read_chunk(chunk_meta: ChunkMeta):
        if chunk_meta.path != read_cache["path"]:
            read_cache["path"] = chunk_meta.path
            read_cache["data"] = read_raw(chunk_meta.path)

        data = read_cache["data"]
        return data.loc[chunk_meta.start : chunk_meta.end]

    for measurement_meta in measurement_metas:
        measurement = pd.concat(list(map(read_chunk, measurement_meta)))
        yield cast(Measurement, measurement)
