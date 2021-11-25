import importlib.resources as resources

import matplotlib.pyplot as plt
import matplotlib as mpl

with resources.path("picarro.resources", "matplotlib-style") as path:
    mpl.style.use(path)

_CONC_COLUMNS = ("CH4", "CO2", "N2O")
_CONC_UNITS = {
    "CH4": "ppmv",
    "CO2": "ppmv",
    "N2O": "ppmv",
}
_SECONDS_PER_MINUTE = 60


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
        figsize=(6.4, 2 + len(columns)),
    )
    elapsed_minutes = (data.index - data.index[0]).seconds / _SECONDS_PER_MINUTE
    for col, ax in zip(columns, axs):
        ax.plot(elapsed_minutes, data[col])
        ax.set_title(_subplot_title(col))

    axs[-1].set_xlabel("Time elapsed (minutes)")

    return fig
