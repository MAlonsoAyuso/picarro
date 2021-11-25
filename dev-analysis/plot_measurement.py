import picarro
from example_data import data_path

data = picarro.read_raw(data_path("example.dat"))
measurements = picarro.split(data)
fig = picarro.plot_measurement(measurements[1])

fig.savefig("outdata/example_data_measurement_1.png")
