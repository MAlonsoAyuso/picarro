import importlib.resources as resources
from typing import Iterable, Mapping, Optional, Sequence
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
from picarro.fluxes import ESTIMATORS, FluxEstimator
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
_MEASUREMENT_KWS = dict(color="k", lw=0, marker=".", markersize=2)


def _subplot_title(column):
    return column


def plot_segment(
    segment: pd.DataFrame,
    columns: Sequence[str],
    flux_estimates: Iterable[FluxEstimator] = (),
    valve_labels: Optional[Mapping[int, str]] = None,
) -> Figure:
    # Rough calculation of height depending on number of panels;
    # nothing scientific at all and probably will break down for large numbers.
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

    (valve_number,) = segment["valve_number"].unique()
    valve_label = (
        valve_labels.get(valve_number, "[missing label]")
        if valve_labels
        else str(valve_number)
    )

    fig.suptitle(f"{valve_label} (valve #{valve_number})")

    segment_start = segment.index[0]

    if len(columns) > 1:
        ax_by_column = dict(zip(columns, axs))  # type: ignore
    else:
        ax_by_column = {columns[0]: axs}

    def calculate_elapsed(time):
        return (time - segment_start).total_seconds() / _SECONDS_PER_MINUTE

    for col in columns:
        ax = ax_by_column[col]
        ax.set_title(_subplot_title(col))
        ax.plot(
            calculate_elapsed(segment.index),
            segment[col],
            **_MEASUREMENT_KWS,
        )

    for flux_estimate in flux_estimates:
        if flux_estimate.moments.data_start != segment_start:
            continue

        moments = flux_estimate.moments
        estimator_times = segment.loc[moments.fit_start : moments.fit_end].index
        assert isinstance(estimator_times, pd.DatetimeIndex)
        estimated_values = flux_estimate.predict(estimator_times)
        ax = ax_by_column[flux_estimate.column]

        # Draw fitted function
        ax.plot(
            calculate_elapsed(estimator_times),
            estimated_values,
            lw=2,
            color=_ESTIMATOR_COLORS[flux_estimate.config.method],
            label=f"{flux_estimate.config.method} fit",
        )

        # Draw vertical line at t0
        ax.axvline(
            [calculate_elapsed(flux_estimate.moments.t0)],
            lw=1,
            color=_ESTIMATOR_COLORS[flux_estimate.config.method],
            linestyle="--",
            label="t0",
        )

    for ax in ax_by_column.values():
        ax.legend(loc="lower left", bbox_to_anchor=(0, 0))

    last_ax = ax_by_column[columns[-1]]
    last_ax.set_xlabel(
        f"Time elapsed (minutes) since\n{segment_start:%Y-%m-%d %H:%M:%S}"
    )

    return fig
