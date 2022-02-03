# Todo

## Definitely
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Write docs
- Decide on a license and update package metadata
- Make parser crash with a nicer message about which file went wrong
- Minor things
  - Check for gaps within files too; cut chunks already when reading file?
- Make overview figure of all files?

## Maybe later

- Relabel measurements using transformation (solenoid#, time span) -> label
- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Allow different output formats (csv, hdf, ...)
- Check goodness-of-fit and warn about bad fit?
- Allow multiple fit methods?
- Add option to filter data based on `species` column in Picarro data files? This could be part of the flux estimation settings, and/or of the measurement/reading settings.
