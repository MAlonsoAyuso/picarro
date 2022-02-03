import importlib.resources as resources
from typing import Iterable, Sequence
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
from picarro.analyze import ESTIMATORS
from picarro.analyze import FluxResult
from picarro.measurements import Measurement
import pandas as pd

# Matplotlib TkAgg backend hogs memory and crashes with too many figures:
# https://github.com/matplotlib/matplotlib/issues/21950
mpl.use("agg")

with resources.path("picarro.resources", "matplotlib-style") as path:
    mpl.style.use(path)  # pyright: reportGeneralTypeIssues=false


_SECONDS_PER_MINUTE = 60

prop_cycle = mpl.rc_params()["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
_ESTIMATOR_COLORS = dict(zip(ESTIMATORS, colors))


def _subplot_title(column):
    return column


def plot_measurement(
    data: Measurement,
    columns: Sequence[str],
    analysis_results: Iterable[FluxResult] = (),
) -> Figure:
    height_per_column = 1.7
    height_extra = 1.3
    height_total = height_per_column * len(columns) + height_extra
    share_extra = height_extra / height_total
    fig, axs = plt.subplots(
        nrows=len(columns),
        sharex=True,
        gridspec_kw=dict(
            top=1 - 0.4 * share_extra,
            hspace=0.3,
            bottom=0.6 * share_extra,
        ),
        figsize=(6.4, height_total),
    )

    measurement_start = data.index[0]

    if len(columns) > 1:
        ax_by_column = dict(zip(columns, axs))  # type: ignore
    else:
        ax_by_column = {columns[0]: axs}

    def calculate_elapsed(time):
        return (time - measurement_start).seconds / _SECONDS_PER_MINUTE

    for col in columns:
        ax = ax_by_column[col]
        ax.set_title(_subplot_title(col))
        ax.plot(calculate_elapsed(data.index), data[col])

    for ar in analysis_results:
        if not ar.measurement_meta.start == measurement_start:
            continue
        if not ar.estimator.column in columns:
            continue

        moments = ar.estimator.moments
        estimator_times = data.loc[moments.fit_start : moments.fit_end].index
        assert isinstance(estimator_times, pd.DatetimeIndex)
        estimated_values = ar.estimator.predict(estimator_times)
        ax = ax_by_column[ar.estimator.column]
        ax.plot(
            calculate_elapsed(estimator_times),
            estimated_values,
            lw=2,
            color=_ESTIMATOR_COLORS[ar.estimator.config.method],
        )

    last_ax = ax_by_column[columns[-1]]
    last_ax.set_xlabel(f"Time elapsed (minutes) since\n{measurement_start}")

    return fig
