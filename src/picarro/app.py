from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import shutil
from typing import Iterator, List
import json

import click
from picarro.core import ConfigProblem

from picarro.fluxes import (
    ESTIMATORS,
    FluxResult,
    FluxEstimator,
    build_fluxes_dataframe,
    estimate_flux,
)
from picarro.config import AppConfig, OutItem
from picarro.measurements import (
    MeasurementMeta,
    read_measurement,
    read_measurements,
)
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


class PicarroPathExists(Exception):
    pass


class PreviousStepRequired(Exception):
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
    logger.debug(f"Preparing write {item} at {path!r}.")
    if path.exists():
        if not config.output.force:
            raise PicarroPathExists(path)
        logger.debug(f"Removing {path!r}.")
        if path.is_file():
            os.remove(path)
        elif path.is_dir():
            shutil.rmtree(path)
        assert not path.exists(), path
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def identify_and_save_measurement_metas(config: AppConfig) -> None:
    path = _prepare_write_path(config, OutItem.measurement_metas_json)
    mms = list(picarro.measurements.identify_measurement_metas(config.measurements))
    obj = _json_converter.unstructure(mms)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_measurement_metas(config: AppConfig) -> List[MeasurementMeta]:
    path = config.output.get_path(OutItem.measurement_metas_json)
    if not path.exists():
        raise PreviousStepRequired("Must identify measurements first.")
    with open(path, "r") as f:
        obj = json.load(f)
    return _json_converter.structure(obj, List[MeasurementMeta])


def export_measurements(config: AppConfig):
    measurement_metas = load_measurement_metas(config)
    out_dir = _prepare_write_path(config, OutItem.measurements_dir)
    out_dir.mkdir()
    cache = {}
    with click.progressbar(
        measurement_metas, show_pos=True, label="Exporting measurements"
    ) as bar:
        for measurement_meta in bar:
            measurement = read_measurement(measurement_meta, config.measurements, cache)
            filename_stem = _build_measurement_file_name_stem(measurement_meta)
            out_path = out_dir / f"{filename_stem}.csv"
            logger.debug(f"Writing measurement to file {out_path!r}.")
            measurement.to_csv(out_path)
    logger.info(f"Saved {len(measurement_metas)} measurement(s) in '{out_dir}'.")


def estimate_and_save_fluxes(config: AppConfig):
    if not config.flux_estimation:
        raise ConfigProblem("No flux estimation config specified.")
    path = _prepare_write_path(config, OutItem.fluxes_json)
    flux_results = list(estimate_fluxes(config))
    _save_flux_results(flux_results, path)


def estimate_fluxes(config: AppConfig) -> Iterator[FluxResult]:
    if not config.flux_estimation:
        raise ConfigProblem("No flux estimation config specified.")
    measurement_metas = load_measurement_metas(config)
    logger.info(
        f"Estimating fluxes ({', '.join(config.flux_estimation.columns)}) "
        f"in {len(measurement_metas)} measurements."
    )
    measurements = read_measurements(measurement_metas, config.measurements)
    for measurement_meta, measurement in zip(measurement_metas, measurements):
        for column in config.flux_estimation.columns:
            series = measurement[column]
            yield FluxResult(
                measurement_meta,
                estimate_flux(config.flux_estimation, series),
            )


def _save_flux_results(flux_results: List[FluxResult], path: Path):
    obj = _json_converter.unstructure(flux_results)
    with open(path, "w") as f:
        json.dump(obj, f)


def _load_flux_results(config: AppConfig) -> List[FluxResult]:
    path = config.output.get_path(OutItem.fluxes_json)
    if not path.exists():
        raise PreviousStepRequired("Must estimate fluxes first.")
    with open(path, "r") as f:
        obj = json.load(f)
    return _json_converter.structure(obj, List[FluxResult])


def export_fluxes_csv(config: AppConfig):
    path = _prepare_write_path(config, OutItem.fluxes_csv)

    flux_results = _load_flux_results(config)

    data = build_fluxes_dataframe(flux_results)
    data.to_csv(path, index=False)
    logger.info(f"Saved {len(data)} flux estimates at '{path}'.")


def _get_flux_results_by_measurement(
    config: AppConfig,
) -> dict[MeasurementMeta, List[FluxResult]]:
    result = defaultdict(list)
    for flux_result in _load_flux_results(config):
        result[flux_result.measurement_meta].append(flux_result)
    return result


def plot_flux_fits(config: AppConfig):
    import matplotlib.pyplot

    flux_results_by_measurement = _get_flux_results_by_measurement(config)
    assert config.flux_estimation
    n_measurements = len(flux_results_by_measurement)

    out_dir = _prepare_write_path(config, OutItem.flux_plots_dir)
    out_dir.mkdir(parents=True)
    cache = {}
    with click.progressbar(
        flux_results_by_measurement.items(), show_pos=True, label="Plotting flux fits"
    ) as bar:
        for measurement_meta, flux_results in bar:
            measurement = read_measurement(measurement_meta, config.measurements, cache)
            fig = picarro.plot.plot_measurement(
                measurement_meta,
                measurement,
                config.flux_estimation.columns,
                flux_results,
            )
            file_name = _build_measurement_file_name_stem(measurement_meta) + ".png"
            path = out_dir / file_name
            fig.savefig(path)
            matplotlib.pyplot.close(fig)
    logger.info(f"Plotted {n_measurements} measurements.")


def _build_measurement_file_name_stem(measurement_meta: MeasurementMeta) -> str:
    date_str = measurement_meta.start.isoformat().replace(":", "_")
    return f"{measurement_meta.valve_label}-{date_str}"
