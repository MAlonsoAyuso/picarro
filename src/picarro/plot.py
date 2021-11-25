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
            bottom=0.12,
        ),
        figsize=(6.4, 2 + len(columns)),
    )
    t0 = data.index[0]
    elapsed_minutes = (data.index - t0).seconds / _SECONDS_PER_MINUTE
    for col, ax in zip(columns, axs):
        ax.plot(elapsed_minutes, data[col])
        ax.set_title(_subplot_title(col))

    axs[-1].set_xlabel(f"Time elapsed (minutes) since\n{t0}")

    return fig
