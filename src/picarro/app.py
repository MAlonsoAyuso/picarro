from __future__ import annotations
from dataclasses import dataclass
import glob
from hashlib import sha256
import itertools
import os
from pathlib import Path
from typing import Any, Callable, Iterator, List
from picarro.analyze import FluxEstimator, estimate_flux
from picarro.config import AppConfig
import picarro.read
from picarro.read import Measurement, MeasurementMeta, ChunkMeta
import pandas as pd
import json
import cattr.preconf.json

_json_converter = cattr.preconf.json.make_converter()
_json_converter.register_unstructure_hook(Path, str)
_json_converter.register_structure_hook(Path, lambda v, _: Path(v))
_json_converter.register_unstructure_hook(pd.Timestamp, str)
_json_converter.register_structure_hook(pd.Timestamp, lambda v, _: pd.Timestamp(v))

_CHUNK_META_DIR = "chunks"


@dataclass
class AnalysisResult:
    measurement_meta: MeasurementMeta
    estimator: FluxEstimator


def iter_measurement_metas(config: AppConfig) -> Iterator[MeasurementMeta]:
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.src_dir)
    for path in file_paths:
        _create_chunk_metas(config, path)

    chunk_metas = itertools.chain(
        *(_load_chunk_metas(config, path) for path in file_paths)
    )

    measurement_metas = picarro.read.iter_measurement_metas(
        chunk_metas,
        config.user.measurements.max_gap,
    )

    for measurement_meta in measurement_metas:
        length = measurement_meta.end - measurement_meta.start  # type: ignore
        min_length = config.user.measurements.min_length
        max_length = config.user.measurements.max_length
        if min_length and length < min_length:
            continue
        if max_length and max_length < length:
            continue
        yield measurement_meta


def iter_measurements(config: AppConfig) -> Iterator[Measurement]:
    return picarro.read.iter_measurements(
        iter_measurement_metas(config), config.columns_to_read
    )


def iter_measurement_pairs(
    config: AppConfig,
) -> Iterator[tuple[MeasurementMeta, Measurement]]:
    mms_1, mms_2 = itertools.tee(iter_measurement_metas(config))
    return zip(mms_1, picarro.read.iter_measurements(mms_2))


def iter_analysis_results(config: AppConfig) -> Iterator[AnalysisResult]:
    for measurement_meta, measurement in iter_measurement_pairs(config):
        for column in config.user.flux_estimation.columns:
            yield AnalysisResult(
                measurement_meta,
                estimate_flux(config.user.flux_estimation, measurement[column]),
            )


def claim_outdir(config: AppConfig) -> Path:
    outdir = config.results_dir_absolute
    marker_file_path = outdir / ".picarro-output"

    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError(
            f"Cannot claim output directory because of existing non-directory {outdir}."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    if not marker_file_path.exists() and list(outdir.iterdir()):
        raise FileExistsError(
            "Cannot claim output directory. "
            f"Expected empty or non-existent directory at {outdir}"
        )

    marker_file_path.touch()

    return outdir


def export_measurements(config: AppConfig):
    outdir = claim_outdir(config) / "measurements"
    outdir.mkdir(exist_ok=True)
    for measurement in iter_measurements(config):
        file_name = measurement.index[0].isoformat().replace(":", "_") + ".csv"
        path = outdir / file_name
        measurement[config.user.measurements.columns].to_csv(path)


def _glob_recursive_in_dir(pattern: str, glob_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    os.chdir(glob_dir)
    result = glob.glob(pattern, recursive=True)
    os.chdir(cwd)
    return [glob_dir / r for r in result]


def _create_chunk_metas(config: AppConfig, data_file_path: Path):
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunk_meta_path(config, data_file_path)
    if meta_path.exists():
        return
    chunk_metas, _ = zip(*picarro.read.iter_chunks(data_file_path))
    _save_json(_json_converter.unstructure(chunk_metas), meta_path)


def _load_chunk_metas(config: AppConfig, data_file_path: Path) -> list[ChunkMeta]:
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunk_meta_path(config, data_file_path)
    data = _load_json(meta_path)
    return _json_converter.structure(data, List[ChunkMeta])


def _get_chunk_meta_path(config: AppConfig, data_file_path: Path) -> Path:
    assert data_file_path.is_absolute(), data_file_path
    file_name = f"{data_file_path.name}-{_repr_hash(data_file_path)}.json"
    return config.cache_dir_absolute / _CHUNK_META_DIR / file_name


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
