# Todo

## Definitely
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Build CLI
- Write docs
- Decide on a license and update package metadata
- Make parser crash with a nicer message about which file went wrong
- Check for infinite value in drop_data_between_valves
- Minor things
  - Make flux estimation config optional
  - Allow list of globs for src
  - Use relative paths in chunk mapping (or absolute if user so specifies)
    - No cache invalidation when simply moving data within computer or to other
  - Use ParsingConfig to determine chunk meta cache dir
  - Let measurements.columns be the list that is exported; check that it is a superset of the columns to be analyzed and/or plotted; and determine the full list to be read as measurements.columns + whatever needed for timestamps, alarms, valves, ...
  - Check for gaps within files too; cut chunks already when reading file?
- Make overview figure of all files?
- Add species selection to measurement mapping
- Generalize from solenoid_valves to any variable of choice

## Maybe later

- Relabel measurements using transformation (solenoid#, time span) -> label
- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Deal with time adjustments? If the Picarro computer has the wrong time setting, it could be nice to be able to apply an offset
- Allow different output formats (csv, hdf, ...)
- Generalize solenoid_valve as indicator of source
