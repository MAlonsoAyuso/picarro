from picarro.chunks import PicarroColumns, read_raw
from picarro.plot import plot_measurement
from picarro.analyze import fit_exponential, fit_line
from example_data import data_path

measurement = read_raw(data_path("example_measurement.dat"))

columns = (
    PicarroColumns.CH4,
    PicarroColumns.CO2,
    PicarroColumns.N2O,
)
A = 0.25  # m2
V = 50e-3  # m3
Q = .25 * 1e-3 / 60  # m3/s
h = V / A  # m
tau = V / Q  # s
t0 = 8 * 60
fig = plot_measurement(
    measurement,
    columns,
    linear_fits={
        col: fit_line(measurement[col], skip_start=10 * 60, skip_end=2.5 * 60)
        for col in columns
    },
    exponential_fits={
        col: fit_exponential(measurement[col], tau=tau, t0=t0, skip_start=10 * 60)
        for col in columns
    },
)

fig.savefig("outdata/example_data_measurement_1.png")
