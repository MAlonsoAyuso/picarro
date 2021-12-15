import datetime
import pandas as pd

EPOCH_TIME_COL = "EPOCH_TIME"
EPOCH_UNIT = "s"


def read_raw(path):
    d = pd.read_csv(path, sep=r"\s+")
    timestamps = d[EPOCH_TIME_COL].pipe(pd.to_datetime, unit=EPOCH_UNIT)
    if not timestamps.is_unique:
        first_duplicate = timestamps.loc[timestamps.duplicated()].iloc[0]
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    d = d.set_index(timestamps)
    return d


def _drop_data_between_valves(data):
    # Column "solenoid_valves" is sometimes noninteger for a short time when switching
    # from one valve to the next. Let's drop these data as they cannot be connected
    # to a chamber.
    valve_num = data["solenoid_valves"]
    is_between_valves = valve_num.astype(int) != valve_num
    return data[~is_between_valves]


def _split_iter(d):
    d = d.pipe(_drop_data_between_valves)
    valve_just_changed = d["solenoid_valves"].diff() != 0
    valve_change_count = valve_just_changed.cumsum()
    for i, g in d.groupby(valve_change_count):
        yield g


def split(d):
    return list(_split_iter(d))
