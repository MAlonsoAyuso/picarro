import picarro

data = picarro.parse("example_data/example.dat")
measurements = picarro.split(data)
fig = picarro.plot_measurement(measurements[1])

fig.savefig("temp.png")
