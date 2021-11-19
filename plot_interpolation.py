from pathlib import Path
import matplotlib.pyplot as plt

import picarro
from picarro import Columns

N_POINTS = 100
SPECIES_NUM = 47
SPECIES_COL = "N2O"
TRANSPARENT = "#00000000"

d = picarro.parse("example_data/example.dat").iloc[:N_POINTS]
is_species_measured = d[Columns.SPECIES] == SPECIES_NUM
linear_interpolation_values = (
    d[SPECIES_COL][is_species_measured]
    .reindex(d.index)
    .interpolate(method="index")[~is_species_measured]
)

fig, ax = plt.subplots()

ax.plot(
    d[SPECIES_COL],
    marker="o",
    ms=5,
    color="#606060",
    mec="#000000",
    mfc=TRANSPARENT,
    label=f"Full Picarro series ({SPECIES_COL})",
)

ax.plot(
    d[SPECIES_COL][is_species_measured],
    ms=5,
    marker="o",
    mec="#000000",
    color="#7070a0",
    lw=0,
    label=f"Picarro values ({SPECIES_COL}) where measured ({Columns.SPECIES} = {SPECIES_NUM})",
)

ax.plot(
    linear_interpolation_values,
    marker="x",
    ms=4,
    lw=0,
    color="#d00000",
    label="Linear interpolation between measurements",
)

ax.legend(loc="lower left", bbox_to_anchor=(0, 1))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

Path("outdata").mkdir(exist_ok=True)
fig.savefig("outdata/interpolation.png", dpi=200, bbox_inches="tight")
