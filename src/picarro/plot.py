import importlib.resources as resources

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from picarro.analyze import fit_line, predict_two_points

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


def plot_measurement(data, columns=_CONC_COLUMNS, fit_line_kws=None):
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

    def calculate_elapsed(time):
        return (time - t0).seconds / _SECONDS_PER_MINUTE

    for col, ax in zip(columns, axs):
        ax.plot(calculate_elapsed(data.index), data[col])
        ax.set_title(_subplot_title(col))

        if fit_line_kws is not None:
            linear_fit = fit_line(data[col], **fit_line_kws)
            line = predict_two_points(linear_fit)
            ax.plot(
                calculate_elapsed(line.index),
                line,
                lw=2,
            )

    axs[-1].set_xlabel(f"Time elapsed (minutes) since\n{t0}")

    return fig
