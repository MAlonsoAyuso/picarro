# Todo

## Definitely
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Write docs
- Decide on a license and update package metadata
- Make parser crash with a nicer message about which file went wrong
- Minor things
  - Let measurements.columns be the list that is exported; check that it is a superset of the columns to be analyzed and/or plotted; and determine the full list to be read as measurements.columns + whatever needed for timestamps, alarms, valves, ...
  - Check for gaps within files too; cut chunks already when reading file?
- Make overview figure of all files?
- Generalize from solenoid_valves to any variable of choice

## Maybe later

- Relabel measurements using transformation (solenoid#, time span) -> label
- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Deal with time adjustments? If the Picarro computer has the wrong time setting, it could be nice to be able to apply an offset
- Allow different output formats (csv, hdf, ...)
- Generalize solenoid_valve as indicator of source
- Check goodness-of-fit and warn about bad fit?
- Allow multiple fit methods?
- Add option to filter data based on `species` column in Picarro data files? This could be part of the flux estimation settings, and/or of the measurement/reading settings.
