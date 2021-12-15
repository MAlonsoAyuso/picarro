# Installing for development

See `CONTRIBUTING.md`

# Using the command-line interface

 ```
 picarro read
 ```

# On the data analysis

## Time and time zones

Many terrible things can happen in data analysis when time is expressed in local time or "wall time", i.e., what is displayed on a clock in a given location. For all the purposes of our data analysis here, it is better to express time in UTC, i.e., a universal, location-independent monotonic time representation.

The UTC time is available in the Picarro data files, expressed as epoch time, in the column `EPOCH_TIME`. For more info on what epoch time is, see https://en.wikipedia.org/wiki/Unix_time

As a general rule, never convert the UTC time representation to anything with time zones unless necessary for comparison to wall clock time or for displaying data in human-readable format.

## `species` numbers

### What is the relation between the `species` column and the concentrations measured?

It seems that at each timestamp, Picarro measures a subset of all the species (N2O, CH4, ...), as indicated by the `species` column in the data files. However, all the species do have values at each timestamp. Based on some plotting and visual inspection, it seems the time series is filled between measurements using linear interpolation.

The correspondence between species numbers and which columns have new values is not obvious. Based on my inspection of the data (2021-11-19), I find the following:
- species 2: H2O and NH3 are updated
- species 25: CH4 is updated
- species 47: CO2 and N2O are updated
  - a complication with species 47 is that they appear (in the one file I have looked at) in pairs, two consecutive timestamps, about one second apart, with only a tiny change in the reported concentrations of CO2 and N2O

### Why this matters to the data analysis

For the purposes of estimating flux rates, not much at all. A linear regression y ~ b0 + b1 t will be only very marginally affected by the introduction of additional points in y through linear interpolation (at least assuming that the points t are somewhat evenly spaced).

*However*, the introduction of artificial data points through interpolation will lead to a decrease in the variance estimates, compared to the "real" underlying series of measurements. Therefore the estimated error bounds of the interpolated time series will be deflated by some amount; in other words, it leads to underestimation of the uncertainty of the gas fluxes.
