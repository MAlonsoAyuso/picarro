import importlib.resources as resources

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from picarro.read import CONC_UNITS

with resources.path("picarro.resources", "matplotlib-style") as path:
    mpl.style.use(path)


_SECONDS_PER_MINUTE = 60


def _subplot_title(column):
    if column in CONC_UNITS:
        return f"{column} ({CONC_UNITS[column]})"

    return column


def plot_measurement(data, columns, linear_fits=None):
    if linear_fits is None:
        linear_fits = {}

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

        if col in linear_fits:
            line = linear_fits[col].predict_endpoints()
            ax.plot(
                calculate_elapsed(line.index),
                line,
                lw=2,
            )

    axs[-1].set_xlabel(f"Time elapsed (minutes) since\n{t0}")

    return fig
