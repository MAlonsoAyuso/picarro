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
- Use pydantic or something to validate configuration file input

## Maybe later

- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Allow different output formats (csv, hdf, ...)
- Allow multiple fit methods?

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
