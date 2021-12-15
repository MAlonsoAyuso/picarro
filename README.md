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
