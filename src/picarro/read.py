import datetime
from os import PathLike
import pandas as pd


class PicarroColumns:
    EPOCH_TIME = "EPOCH_TIME"
    solenoid_valves = "solenoid_valves"


class ChunkMetaColumns:
    start = "start"
    end = "end"
    solenoid_valve = "solenoid_valve"


def read_raw(path):
    d = pd.read_csv(path, sep=r"\s+").pipe(_reindex_timestamp)
    return d


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


def iter_chunks(d):
    d = d.pipe(_drop_data_between_valves)
    valve_just_changed = d[PicarroColumns.solenoid_valves].diff() != 0
    valve_change_count = valve_just_changed.cumsum()
    for i, g in d.groupby(valve_change_count):
        yield g


def _drop_data_between_valves(data):
    # Column "solenoid_valves" is sometimes noninteger for a short time when switching
    # from one valve to the next. Let's drop these data as they cannot be connected
    # to a chamber.
    valve_num = data[PicarroColumns.solenoid_valves]
    is_between_valves = valve_num.astype(int) != valve_num
    return data[~is_between_valves].astype({PicarroColumns.solenoid_valves: int})


def _get_chunk_metadata(chunk):
    solenoid_valves = chunk[PicarroColumns.solenoid_valves].unique()
    assert len(solenoid_valves) == 1, solenoid_valves
    (the_valve,) = solenoid_valves
    return {
        ChunkMetaColumns.start: chunk.index[0],
        ChunkMetaColumns.end: chunk.index[-1],
        ChunkMetaColumns.solenoid_valve: the_valve,
    }


def get_chunks_metadata(d):
    return pd.DataFrame(list(map(_get_chunk_metadata, iter_chunks(d))))


def write_chunks_metadata(chunks_metadata, path: PathLike):
    chunks_metadata.to_csv(path, index=False)


def read_chunks_metadata(path: PathLike):
    return pd.read_csv(
        path,
        parse_dates=[ChunkMetaColumns.start, ChunkMetaColumns.end],
        dtype={
            ChunkMetaColumns.solenoid_valve: int,
        },
    )
