# Todo

## Definitely
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Build CLI
- Write docs
- Decide on a license and update package metadata
- Minor things
  - Make flux estimation config optional

## Maybe later

- Relabel measurements using transformation (solenoid#, time span) -> label
- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Deal with time adjustments? If the Picarro computer has the wrong time setting, it could be nice to be able to apply an offset
- Allow different output formats (csv, hdf, ...)
- Generalize solenoid_valve as indicator of source
