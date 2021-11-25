from pathlib import Path
import matplotlib.pyplot as plt

import picarro
from example_data import data_path


I_START = 200
N_POINTS = 50
TRANSPARENT = "#00000000"


def plot_interpolation(species_num, species_col, ax):
    d = picarro.read_raw(data_path("example.dat")).iloc[I_START : I_START + N_POINTS]

    is_species_measured = d["species"] == species_num
    linear_interpolation_values = (
        d[species_col][is_species_measured]
        .reindex(d.index)
        .interpolate(method="index", limit_area="inside")[~is_species_measured]
    )

    ax.plot(
        d[species_col],
        marker="o",
        ms=5,
        color="#606060",
        mec="#000000",
        mfc=TRANSPARENT,
        label=f"Full Picarro series",
    )

    ax.plot(
        d[species_col][is_species_measured],
        ms=5,
        marker="o",
        mec="#000000",
        color="#5050a0",
        lw=0,
        label=f"Picarro values where measured (species column matches)",
    )

    ax.plot(
        linear_interpolation_values,
        marker="x",
        ms=2,
        lw=0,
        color="#d00000",
        label="Linear interpolation between measurements",
    )

    ax.set_title(f'species = {species_num}, column "{species_col}"')

    ax.grid(True, axis="x")

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


PAIRS = [
    dict(species_num=2, species_col="H2O"),
    dict(species_num=2, species_col="NH3"),
    dict(species_num=25, species_col="CH4"),
    dict(species_num=47, species_col="CO2"),
    dict(species_num=47, species_col="N2O"),
]

fig, axs = plt.subplots(nrows=len(PAIRS), sharex=True, figsize=(6, 8))

for pair, ax in zip(PAIRS, axs):
    plot_interpolation(**pair, ax=ax)


axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.2))

Path("outdata").mkdir(exist_ok=True)
fig.set_tight_layout(True)
fig.savefig("outdata/interpolation.png", dpi=200, bbox_inches="tight")
