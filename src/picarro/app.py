from __future__ import annotations
import glob
import itertools
import os
from pathlib import Path
from typing import Iterator
from picarro.config import AppConfig
import picarro.read
import pandas as pd

_CONFIG_TIME_UNIT = "s"


def iter_measurements(config: AppConfig) -> Iterator[picarro.read.Measurement]:
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.src_dir)
    chunks_meta = (
        picarro.read.get_chunks_metadata(picarro.read.read_raw(p), p)
        for p in file_paths
    )
    measurements_meta = picarro.read.iter_measurements_meta(
        itertools.chain(*chunks_meta),
        pd.Timedelta(config.user.measurements.max_gap, _CONFIG_TIME_UNIT),
    )
    return picarro.read.iter_measurements(measurements_meta)


def _glob_recursive_in_dir(pattern: str, glob_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    os.chdir(glob_dir)
    result = glob.glob(pattern, recursive=True)
    os.chdir(cwd)
    return [glob_dir / r for r in result]
