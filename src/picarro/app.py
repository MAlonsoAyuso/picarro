from __future__ import annotations
import glob
import itertools
import os
from pathlib import Path
from typing import Any, Iterator, List
import functools
from picarro import measurements
from picarro.analyze import AnalysisResult, estimate_flux
from picarro.config import AppConfig
from picarro.measurements import Measurement, MeasurementMeta
import picarro.measurements
import picarro.chunks
import picarro.plot
from picarro.chunks import ChunkMeta, get_chunk_metas
import pandas as pd
import logging.config
import logging

logger = logging.getLogger(__name__)


def log_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
            raise

    return wrapper


@log_unhandled_exceptions
def export_measurements(config: AppConfig):
    claim_outdir(config)
    config.paths.out_measurements.mkdir(exist_ok=True)
    count = 0
    for measurement in iter_measurements(config):
        count += 1
        filename_stem = _build_measurement_filename_stem(measurement)
        out_path = config.paths.out_measurements / f"{filename_stem}.csv"
        logger.debug(f"Writing measurement to file {out_path}.")
        measurement[config.user.measurements.columns].to_csv(out_path)
    logger.info(f"Wrote {count} measurements to files.")


@log_unhandled_exceptions
def export_fluxes(config: AppConfig):
    claim_outdir(config)
    config.paths.out_measurements.mkdir(exist_ok=True)
    data = _get_fluxes_dataframe(config)
    data.to_csv(config.paths.out_fluxes, index=False)
    logger.info(f"Wrote fluxes from {len(data)} measurements.")


@log_unhandled_exceptions
def plot_fluxes(config: AppConfig):
    claim_outdir(config)
    analysis_results = list(_iter_analysis_results(config))
    count = 0
    for measurement in iter_measurements(config):
        count += 1
        fig = picarro.plot.plot_measurement(
            measurement, config.user.flux_estimation.columns, analysis_results
        )
        file_name = _build_measurement_filename_stem(measurement) + ".png"
        path = config.paths.out_plot_fluxes / file_name
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(path)
    logger.info(f"Plotted {count} measurements.")


def setup_logging(config: AppConfig):
    cwd = Path.cwd()
    os.chdir(config.paths.out)
    logging.config.dictConfig(config.user.logging)
    os.chdir(cwd)


def _iter_measurement_pairs(
    config: AppConfig,
) -> Iterator[tuple[MeasurementMeta, Measurement]]:
    mms_1, mms_2 = itertools.tee(_iter_measurement_metas(config))
    return zip(
        mms_1, picarro.measurements.read_measurements(mms_2, config.user.measurements)
    )


def iter_measurements(config: AppConfig) -> Iterator[Measurement]:
    for _, measurement in _iter_measurement_pairs(config):
        yield measurement


def _iter_all_chunk_metas(config: AppConfig) -> Iterator[ChunkMeta]:
    claim_outdir(config)
    file_paths = _glob_recursive_in_dir(config.user.measurements.src, config.base_dir)

    file_count = 0
    chunk_count = 0
    for path in file_paths:
        file_count += 1
        for chunk in get_chunk_metas(
            path, config.user.measurements, cache_dir=config.paths.cache_chunks
        ):
            chunk_count += 1
            yield chunk
    logger.info(f"Read {chunk_count} chunks from {file_count} files.")


def _iter_measurement_metas(config: AppConfig) -> Iterator[MeasurementMeta]:
    chunk_metas = _iter_all_chunk_metas(config)
    return picarro.measurements.stitch_chunk_metas(
        chunk_metas, config.user.measurements
    )


def _get_fluxes_dataframe(config: AppConfig) -> pd.DataFrame:
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

    rows = list(map(make_row, _iter_analysis_results(config)))
    return pd.DataFrame(rows)


def _iter_analysis_results(config: AppConfig) -> Iterator[AnalysisResult]:
    for measurement_meta, measurement in _iter_measurement_pairs(config):
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
        logger.debug(f"Outdir contains {list(outdir.iterdir())}")
        raise FileExistsError(
            "Cannot claim output directory. "
            f"Expected empty or non-existent directory at {outdir}"
        )

    marker_file_path.touch()


def _glob_recursive_in_dir(pattern: str, glob_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    os.chdir(glob_dir)
    result = glob.glob(pattern, recursive=True)
    os.chdir(cwd)
    return [glob_dir / r for r in result]


def _build_measurement_filename_stem(measurement: Measurement) -> str:
    return measurement.index[0].isoformat().replace(":", "_")
