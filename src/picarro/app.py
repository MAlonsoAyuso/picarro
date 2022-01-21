from __future__ import annotations
import glob
import itertools
import os
from pathlib import Path
from typing import Any, Iterator, List
from picarro.analyze import AnalysisResult, estimate_flux
from picarro.config import AppConfig
import picarro.read
import picarro.plot
from picarro.read import Measurement, MeasurementMeta, ChunkMeta
import pandas as pd
import json
import cattr.preconf.json

_json_converter = cattr.preconf.json.make_converter()
_json_converter.register_unstructure_hook(Path, str)
_json_converter.register_structure_hook(Path, lambda v, _: Path(v))
_json_converter.register_unstructure_hook(pd.Timestamp, str)
_json_converter.register_structure_hook(pd.Timestamp, lambda v, _: pd.Timestamp(v))


def iter_measurement_metas(config: AppConfig) -> Iterator[MeasurementMeta]:
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.base_dir)
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
        duration = measurement_meta.end - measurement_meta.start  # type: ignore
        min_duration = config.user.measurements.min_duration
        max_duration = config.user.measurements.max_duration
        if min_duration and duration < min_duration:
            continue
        if max_duration and max_duration < duration:
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


def claim_outdir(config: AppConfig):
    outdir = config.paths.out
    marker_file_path = config.paths.out_marker

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


def _build_measurement_filename_stem(measurement: Measurement) -> str:
    return measurement.index[0].isoformat().replace(":", "_")


def export_measurements(config: AppConfig):
    claim_outdir(config)
    config.paths.out_measurements.mkdir(exist_ok=True)
    for measurement in iter_measurements(config):
        file_name = _build_measurement_filename_stem(measurement) + ".csv"
        path = config.paths.out_measurements / file_name
        measurement[config.user.measurements.columns].to_csv(path)


def get_fluxes_dataframe(config: AppConfig) -> pd.DataFrame:
    def make_row(analysis_result: AnalysisResult):
        return pd.Series(
            dict(
                start_utc=analysis_result.measurement_meta.start,
                end_utc=analysis_result.measurement_meta.end,
                solenoid_valve=analysis_result.measurement_meta.solenoid_valve,
                column=analysis_result.estimator.column,
                vol_flux=analysis_result.estimator.estimate_vol_flux(),
                n_samples_total=analysis_result.measurement_meta.n_samples,
                n_samples_flux_estimate=analysis_result.estimator.n_samples,
            )
        )

    rows = list(map(make_row, iter_analysis_results(config)))
    return pd.DataFrame(rows)


def export_fluxes(config: AppConfig):
    claim_outdir(config)
    config.paths.out_measurements.mkdir(exist_ok=True)
    data = get_fluxes_dataframe(config)
    data.to_csv(config.paths.out_fluxes, index=False)


def plot_fluxes(config: AppConfig):
    claim_outdir(config)
    analysis_results = list(iter_analysis_results(config))
    for measurement in iter_measurements(config):
        fig = picarro.plot.plot_measurement(
            measurement, config.user.flux_estimation.columns, analysis_results
        )
        file_name = _build_measurement_filename_stem(measurement) + ".png"
        path = config.paths.out_plot_fluxes / file_name
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(path)


def _glob_recursive_in_dir(pattern: str, glob_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    os.chdir(glob_dir)
    result = glob.glob(pattern, recursive=True)
    os.chdir(cwd)
    return [glob_dir / r for r in result]


def _create_chunk_metas(config: AppConfig, data_file_path: Path):
    claim_outdir(config)
    assert data_file_path.is_absolute(), data_file_path
    meta_path = config.paths.cache_chunk_meta(data_file_path)
    if meta_path.exists():
        return
    chunk_metas, _ = zip(*picarro.read.iter_chunks(data_file_path))
    _save_json(_json_converter.unstructure(chunk_metas), meta_path)


def _load_chunk_metas(config: AppConfig, data_file_path: Path) -> list[ChunkMeta]:
    assert data_file_path.is_absolute(), data_file_path
    meta_path = config.paths.cache_chunk_meta(data_file_path)
    data = _load_json(meta_path)
    return _json_converter.structure(data, List[ChunkMeta])


def _save_json(obj: Any, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=2))


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.loads(f.read())
