from picarro.read import PicarroColumns, read_raw
from picarro.plot import plot_measurement
from picarro.analyze import fit_line
from example_data import data_path

measurement = read_raw(data_path("example_measurement.dat"))

columns = (
    PicarroColumns.CH4,
    PicarroColumns.CO2,
    PicarroColumns.N2O,
)
fig = plot_measurement(
    measurement,
    columns,
    linear_fits={
        col: fit_line(measurement[col], skip_start=10 * 60, skip_end=2.5 * 60)
        for col in columns
    },
)

fig.savefig("outdata/example_data_measurement_1.png")
