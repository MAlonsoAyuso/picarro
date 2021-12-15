# Todo

## Definitely
- Stitch together chunks from different files -> measurements
- Check that measurements are OK, e.g., no time gaps, no alarm, ...
- Read a list of data files as a common file
  - Find all data files
  - Map the chunks in each file
  - Stitch the chunks
  - Generate measurements = stitched chunks without gaps
- Build CLI
- Write docs
- Decide on a license and update package metadata

## Maybe later

- Relabel measurements using transformation (solenoid#, time span) -> label
- Be more permissive about versions of dependencies (currently requiring latest version of pandas, matplotlib, scipy, click, ...)
- Deal with time adjustments? If the Picarro computer has the wrong time setting, it could be nice to be able to apply an offset
- Allow different output formats (csv, hdf, ...)
