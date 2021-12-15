from picarro.read import read_raw, iter_chunks
from picarro.plot import plot_measurement
from example_data import data_path

data = read_raw(data_path("example.dat"))
measurements = list(iter_chunks(data))
measurement = measurements[1]
fig = plot_measurement(
    measurement,
    fit_line_kws=dict(skip_start=60 * 10, skip_end=60 * 2.5),
)

fig.savefig("outdata/example_data_measurement_1.png")
