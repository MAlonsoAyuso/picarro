from __future__ import annotations
import glob
from hashlib import sha256
import itertools
import os
from pathlib import Path
from typing import Any, Iterator
from picarro.config import AppConfig
import picarro.read
import pandas as pd

_CONFIG_TIME_UNIT = "s"
_CHUNKS_META_DIR = "chunks"


def iter_measurements(config: AppConfig) -> Iterator[picarro.read.Measurement]:
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.src_dir)
    for path in file_paths:
        _create_chunks_meta(config, path)

    chunks_meta = (_load_chunks_meta(config, path) for path in file_paths)

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


def _create_chunks_meta(config: AppConfig, data_file_path: Path):
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunks_meta_path(config, data_file_path)
    if meta_path.exists():
        return
    data = picarro.read.read_raw(data_file_path)
    chunks_meta = picarro.read.get_chunks_metadata(data, data_file_path)
    picarro.read.save_chunks_meta(chunks_meta, meta_path)


def _load_chunks_meta(
    config: AppConfig, data_file_path: Path
) -> list[picarro.read.ChunkMeta]:
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunks_meta_path(config, data_file_path)
    return picarro.read.load_chunks_meta(meta_path)


def _get_chunks_meta_path(config: AppConfig, data_file_path: Path) -> Path:
    assert data_file_path.is_absolute(), data_file_path
    file_name = f"{data_file_path.name}-{_repr_hash(data_file_path)}.json"
    return config.cache_dir_absolute / _CHUNKS_META_DIR / file_name


def _repr_hash(obj: Any) -> str:
    m = sha256()
    m.update(repr(obj).encode())
    return m.hexdigest()
