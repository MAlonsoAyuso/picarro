from __future__ import annotations
from dataclasses import dataclass
import glob
from hashlib import sha256
import itertools
import os
from pathlib import Path
from typing import Any, Iterator, List
from picarro.analyze import FluxEstimator
from picarro.config import AppConfig
import picarro.read
from picarro.read import Measurement, MeasurementMeta, ChunkMeta
import pandas as pd
import json
import cattr.preconf.json

_json_converter = cattr.preconf.json.make_converter()
_json_converter.register_unstructure_hook(pd.Timestamp, str)
_json_converter.register_structure_hook(pd.Timestamp, lambda v, _: pd.Timestamp(v))

_CONFIG_TIME_UNIT = "s"
_CHUNKS_META_DIR = "chunks"


@dataclass
class AnalysisResult:
    measurement: MeasurementMeta
    estimator: FluxEstimator


def iter_measurements_meta(config: AppConfig) -> Iterator[MeasurementMeta]:
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.src_dir)
    for path in file_paths:
        _create_chunks_meta(config, path)

    chunks_meta = (_load_chunks_meta(config, path) for path in file_paths)

    measurement_metas = picarro.read.iter_measurements_meta(
        itertools.chain(*chunks_meta),
        pd.Timedelta(config.user.measurements.max_gap, _CONFIG_TIME_UNIT),
    )

    for measurement_meta in measurement_metas:
        length = measurement_meta.end - measurement_meta.start  # type: ignore
        min_length = pd.Timedelta(
            config.user.measurements.min_length, _CONFIG_TIME_UNIT
        )
        max_length = pd.Timedelta(
            config.user.measurements.max_length, _CONFIG_TIME_UNIT
        )
        if length < min_length:
            continue
        if max_length < length:
            continue
        yield measurement_meta


def iter_measurements(config: AppConfig) -> Iterator[Measurement]:
    measurements_meta = iter_measurements_meta(config)
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
    _save_json(_json_converter.unstructure(chunks_meta), meta_path)


def _load_chunks_meta(config: AppConfig, data_file_path: Path) -> list[ChunkMeta]:
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunks_meta_path(config, data_file_path)
    data = _load_json(meta_path)
    return _json_converter.structure(data, List[ChunkMeta])


def _get_chunks_meta_path(config: AppConfig, data_file_path: Path) -> Path:
    assert data_file_path.is_absolute(), data_file_path
    file_name = f"{data_file_path.name}-{_repr_hash(data_file_path)}.json"
    return config.cache_dir_absolute / _CHUNKS_META_DIR / file_name


def _repr_hash(obj: Any) -> str:
    m = sha256()
    m.update(repr(obj).encode())
    return m.hexdigest()


def _save_json(obj: Any, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=2))


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.loads(f.read())
