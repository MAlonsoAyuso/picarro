# Todo

## Definitely
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Write docs
- Decide on a license and update package metadata
- Make parser crash with a nicer message about which file went wrong
- Make overview figure of all files?
- Allow filtering of data based on
  - Alarm
  - Concentrations
  - `species`
  - ...
  - maybe more generally, possibility to make a list of columns and allowed min/max values or list of allowed values; could be applied on the row level
- Add goodness-of-fit measure to flux estimates info
- Allow import and analysis commands to incrementally update when there are new data; maybe a general `--fill` flag? Maybe have `--fill` as the default?
- Allow commands to be called only for info? Maybe almost the same as doing `--fill`?
- Make a nice command to do "everything", i.e., all that is listed in the config file, in the correct order. Should also be possible with both `--fill` and `--overwrite` or `--force` or whatever we call it.
- get valve labels back

## Maybe later

- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Allow different output formats (csv, hdf, ...)
- Allow multiple fit methods?
- Allow removal of measurement data but keeping the flux estimates etc; this would make it easier to send the sqlite database by email etc

# Structure for a new program using an Sqlite database instead

ensure that data files are non-overlapping

read picarro files
- parse timestamps
- exclude unwanted cols based on name
- drop rows with noninteger valve number

put into an SQLite database

filter based on column values
- exclude unwanted rows, e.g., with alarm etc

determine segments based on filtered data

analyze segments
