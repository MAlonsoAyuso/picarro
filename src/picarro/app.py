from __future__ import annotations
import glob
import itertools
import os
from pathlib import Path
import shutil
from typing import Iterator, List, Sequence
import json

from picarro.analyze import (
    ESTIMATORS,
    FluxResult,
    FluxEstimator,
    estimate_flux,
)
from picarro.config import AppConfig, OutItem
from picarro.measurements import Measurement, MeasurementMeta
import picarro.measurements
import picarro.chunks
import picarro.plot
import pandas as pd
import logging.config
import logging
import cattr.preconf.json

logger = logging.getLogger(__name__)


_json_converter = cattr.preconf.json.make_converter()
_json_converter.register_unstructure_hook(Path, Path.as_posix)
_json_converter.register_structure_hook(Path, lambda v, _: Path(v))
_json_converter.register_unstructure_hook(pd.Timestamp, str)
_json_converter.register_structure_hook(pd.Timestamp, lambda v, _: pd.Timestamp(v))
_json_converter.register_unstructure_hook(pd.Timedelta, str)
_json_converter.register_structure_hook(pd.Timedelta, lambda v, _: pd.Timedelta(v))


_json_converter.register_structure_hook(
    FluxEstimator,
    lambda obj, _: _json_converter.structure(obj, ESTIMATORS[obj["config"]["method"]]),
)


class ConfigError(FileExistsError):
    pass


class PicarroPathExists(FileExistsError):
    pass


class PreviousStepRequired(RuntimeError):
    pass


def setup_logging(config: AppConfig):
    logging_dir = config.output.out_dir
    cwd = Path.cwd()
    logging_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(logging_dir)
    logging.config.dictConfig(config.logging)
    os.chdir(cwd)


def _prepare_write_path(config: AppConfig, item: OutItem) -> Path:
    path = config.output.get_path(item)
    logger.debug(f"Preparing write {item} at {path}")
    if path.exists():
        if not config.output.force:
            raise PicarroPathExists(path)
        logger.debug(f"Removing {path}")
        if path.is_file():
            os.remove(path)
        elif path.is_dir():
            shutil.rmtree(path)
        assert not path.exists(), path
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def identify_measurements(config: AppConfig) -> None:
    path = _prepare_write_path(config, OutItem.measurement_metas_json)
    chunk_metas = _iter_chunk_metas(config)
    measurement_metas = list(
        picarro.measurements.stitch_chunk_metas(chunk_metas, config.measurements)
    )
    _save_measurement_metas(measurement_metas, path)


def _save_measurement_metas(measurement_metas: Sequence[MeasurementMeta], path: Path):
    obj = _json_converter.unstructure(measurement_metas)
    with open(path, "w") as f:
        json.dump(obj, f)


def _iter_chunk_metas(config: AppConfig) -> Iterator[picarro.chunks.ChunkMeta]:
    glob_patterns = config.measurements.src
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    for glob_pattern in glob_patterns:
        file_paths = list(map(Path, glob.glob(glob_pattern, recursive=True)))
        logger.info(
            f"Found {len(file_paths)} source files using pattern {glob_pattern}"
        )
        for path in file_paths:
            yield from picarro.chunks.read_chunks(path, config.measurements)


def load_measurement_metas(config: AppConfig) -> List[MeasurementMeta]:
    path = config.output.get_path(OutItem.measurement_metas_json)
    if not path.exists():
        raise PreviousStepRequired(
            "Cannot load measurements before analyzing input data."
        )
    with open(path, "r") as f:
        obj = json.load(f)
    return _json_converter.structure(obj, List[MeasurementMeta])


def export_measurements(config: AppConfig):
    measurement_metas = load_measurement_metas(config)
    out_dir = _prepare_write_path(config, OutItem.measurements_dir)
    out_dir.mkdir()
    measurements = picarro.measurements.read_measurements(
        measurement_metas, config.measurements
    )
    for measurement in measurements:
        filename_stem = _build_measurement_file_name_stem(measurement)
        out_path = out_dir / f"{filename_stem}.csv"
        logger.debug(f"Writing measurement to file {out_path}.")
        measurement[config.measurements.columns].to_csv(out_path)
    logger.info(f"Wrote {len(measurement_metas)} measurement(s) to files.")


def estimate_fluxes(config: AppConfig):
    if not config.flux_estimation:
        raise ConfigError("No flux estimation config specified.")
    path = _prepare_write_path(config, OutItem.fluxes_json)
    analysis_results = list(analyze_fluxes(config))
    columns = {ar.estimator.column for ar in analysis_results}
    n_measurements = len({ar.measurement_meta for ar in analysis_results})
    _save_analysis_results(analysis_results, path)
    logger.info(
        f"Estimated {len(analysis_results)} fluxes ({', '.join(columns)}) "
        f"in {n_measurements} measurements."
    )


def _iter_measurement_pairs(
    config: AppConfig,
) -> Iterator[tuple[MeasurementMeta, Measurement]]:
    mms = load_measurement_metas(config)
    mms_1, mms_2 = itertools.tee(mms)
    return zip(
        mms_1, picarro.measurements.read_measurements(mms_2, config.measurements)
    )


def analyze_fluxes(config: AppConfig) -> Iterator[FluxResult]:
    assert config.flux_estimation
    for measurement_meta, measurement in _iter_measurement_pairs(config):
        for column in config.flux_estimation.columns:
            series = measurement[column]
            yield FluxResult(
                measurement_meta,
                estimate_flux(config.flux_estimation, series),
            )


def _save_analysis_results(analysis_results: List[FluxResult], path: Path):
    obj = _json_converter.unstructure(analysis_results)
    with open(path, "w") as f:
        json.dump(obj, f)


def _load_analysis_results(config: AppConfig) -> List[FluxResult]:
    path = config.output.get_path(OutItem.fluxes_json)
    if not path.exists():
        raise PreviousStepRequired("Cannot load flux analyses before analyzing.")
    with open(path, "r") as f:
        obj = json.load(f)
    return _json_converter.structure(obj, List[FluxResult])


def export_fluxes_csv(config: AppConfig):
    path = _prepare_write_path(config, OutItem.fluxes_csv)

    analysis_results = _load_analysis_results(config)

    data = _build_fluxes_dataframe(analysis_results)
    data.to_csv(path, index=False)
    logger.info(f"Saved results at {path}")


def _build_fluxes_dataframe(analysis_results: List[FluxResult]) -> pd.DataFrame:
    def make_row(analysis_result: FluxResult):
        return pd.Series(
            dict(
                start_utc=analysis_result.measurement_meta.start,
                end_utc=analysis_result.measurement_meta.end,
                valve_number=analysis_result.measurement_meta.valve_number,
                column=analysis_result.estimator.column,
                vol_flux=analysis_result.estimator.estimate_vol_flux(),
                n_samples_total=analysis_result.measurement_meta.n_samples,
                n_samples_flux_estimate=analysis_result.estimator.n_samples,
            )
        )

    rows = list(map(make_row, analysis_results))
    return pd.DataFrame(rows)


def plot_flux_fits(config: AppConfig):
    import matplotlib.pyplot

    analysis_results = _load_analysis_results(config)
    assert config.flux_estimation
    measurement_metas = load_measurement_metas(config)
    measurements = picarro.measurements.read_measurements(
        measurement_metas, config.measurements
    )
    n_measurements = len(measurement_metas)

    out_dir = _prepare_write_path(config, OutItem.flux_plots_dir)
    out_dir.mkdir(parents=True)
    for measurement in measurements:
        fig = picarro.plot.plot_measurement(
            measurement, config.flux_estimation.columns, analysis_results
        )
        file_name = _build_measurement_file_name_stem(measurement) + ".png"
        path = out_dir / file_name
        fig.savefig(path)
        matplotlib.pyplot.close(fig)
    logger.info(f"Plotted {n_measurements} measurements.")


def _build_measurement_file_name_stem(measurement: Measurement) -> str:
    return measurement.index[0].isoformat().replace(":", "_")
