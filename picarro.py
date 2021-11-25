import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("plot-style")

_TIMESTAMP_FORMAT = r"%Y-%m-%d %H:%M:%S.%f"
_CONC_COLUMNS = ("CH4", "CO2", "N2O")
_CONC_UNITS = {
    "CH4": "ppmv",
    "CO2": "ppmv",
    "N2O": "ppmv",
}
_SECONDS_PER_MINUTE = 60


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


def _subplot_title(column):
    if column in _CONC_UNITS:
        return f"{column} ({_CONC_UNITS[column]})"
    
    return column


def plot_measurement(data, columns=_CONC_COLUMNS):
    fig, axs = plt.subplots(
        nrows=len(columns),
        sharex=True,
        gridspec_kw=dict(
            hspace=0.4,
        ),
        figsize=(6.4, 2+len(columns))
    )
    elapsed_minutes = (data.index - data.index[0]).seconds / _SECONDS_PER_MINUTE
    for col, ax in zip(columns, axs):
        ax.plot(elapsed_minutes, data[col])
        ax.set_title(_subplot_title(col))
    
    axs[-1].set_xlabel("Time elapsed (minutes)")

    return fig
