import datetime
import pandas as pd

_TIMESTAMP_FORMAT = r"%Y-%m-%d %H:%M:%S.%f"


def parse(path):
    d = pd.read_csv(path, sep=r"\s+")
    timestamps = _parse_timestamp(d)
    if not timestamps.is_unique:
        first_duplicate = timestamps[timestamps.duplicated()].iloc[0]
        raise ValueError(f"non-unique timestamp {first_duplicate}")
    d.index = timestamps
    return d


def _parse_timestamp(d):
    timestamp_str = d["DATE"] + " " + d["TIME"]
    return timestamp_str.apply(datetime.datetime.strptime, args=(_TIMESTAMP_FORMAT,))


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
