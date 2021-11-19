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


class Columns:
    SPECIES = "species"
