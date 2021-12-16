from os import PathLike
from typing import Iterable, Iterator, NewType, cast
import pandas as pd


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

# ChunkMeta: A DataFrame naming start and end times for a chunk, and the active solenoid
ChunkMeta = NewType("ChunkMeta", pd.DataFrame)

# A combination of information from ChunkMetas and Picarro data file paths
ChunkMap = NewType("ChunkMap", pd.DataFrame)


class ChunkMetaColumns:
    start = "start"
    end = "end"
    solenoid_valve = "solenoid_valve"


def read_raw(path: PathLike) -> DataFile:
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
        unit="ms",
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


def _get_chunk_metadata(chunk: Chunk):
    solenoid_valves = chunk[PicarroColumns.solenoid_valves].unique()
    assert len(solenoid_valves) == 1, solenoid_valves
    (the_valve,) = solenoid_valves
    return {
        ChunkMetaColumns.start: chunk.index[0],
        ChunkMetaColumns.end: chunk.index[-1],
        ChunkMetaColumns.solenoid_valve: the_valve,
    }


def get_chunks_metadata(d: DataFile) -> ChunkMeta:
    result = pd.DataFrame(list(map(_get_chunk_metadata, iter_chunks(d))))
    return cast(ChunkMeta, result)


def write_chunks_metadata(chunks_metadata, path: PathLike):
    chunks_metadata.to_csv(path, index=False)


def read_chunks_metadata(path: PathLike) -> ChunkMeta:
    result = pd.read_csv(
        path,
        parse_dates=[ChunkMetaColumns.start, ChunkMetaColumns.end],
        dtype={
            ChunkMetaColumns.solenoid_valve: int,
        },
    )
    return cast(ChunkMeta, result)


def build_chunk_map(paths: Iterable[PathLike]) -> ChunkMap:
    chunk_map = (
        pd.concat(
            {path: get_chunks_metadata(read_raw(path)) for path in paths},
            names=["path", "chunk_index"],
        )
        .reset_index()
        .sort_values(ChunkMetaColumns.start)
    )

    return cast(ChunkMap, chunk_map)


def _check_no_overlaps(measurement_map):
    overlapping = measurement_map["time_gap_from_previous"] < pd.Timedelta(0)  # type: ignore
    if overlapping.any():
        violator_index = overlapping[overlapping].index[0]
        violators = measurement_map.loc[violator_index - 1 : violator_index]
        raise ValueError(f"overlapping chunks: {violators}")


def iter_measurements(
    chunk_map: ChunkMap, max_gap: pd.Timedelta = DEFAULT_MAX_GAP_BETWEEN_CHUNKS
) -> Iterator[Measurement]:
    measurement_map = chunk_map.assign(
        time_gap_from_previous=lambda d: (
            d[ChunkMetaColumns.start] - d[ChunkMetaColumns.end].shift(1)
        ),
        is_adjacent_to_previous=lambda d: (d["time_gap_from_previous"] < max_gap),
        same_valve_as_previous=lambda d: (
            d[ChunkMetaColumns.solenoid_valve].diff() == 0
        ),
        chunk_starts_new_measurement=lambda d: (
            ~(d["same_valve_as_previous"] & d["is_adjacent_to_previous"])
        ).fillna(
            True
        ),  # fillna(True) needed for the first row
        measurement_number=lambda d: d["chunk_starts_new_measurement"].cumsum(),
    )
    _check_no_overlaps(measurement_map)

    read_cache = {
        "path": None,
        "chunks": [],
    }
    def read_chunk(path, chunk_index):
        if path == read_cache["path"]:
            chunks = read_cache["chunks"]
        else:
            chunks = list(iter_chunks(read_raw(path)))
            read_cache["path"] = path
            read_cache["chunks"] = chunks

        return chunks[chunk_index]

    for _, g in measurement_map.groupby("measurement_number"):
        measurement = pd.concat(
            [read_chunk(row["path"], row["chunk_index"]) for _, row in g.iterrows()]
        )
        yield cast(Measurement, measurement)
