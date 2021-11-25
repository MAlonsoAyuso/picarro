import picarro
from example_data import data_path

data = picarro.read_raw(data_path("example.dat"))
measurements = picarro.split(data)
measurement = measurements[1]
fig = picarro.plot_measurement(
    measurement,
    fit_line_kws=dict(skip_start=60 * 10, skip_end=60 * 2.5),
)

fig.savefig("outdata/example_data_measurement_1.png")
