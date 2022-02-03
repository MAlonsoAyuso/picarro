from __future__ import annotations
from collections import defaultdict
import os
from pathlib import Path
import shutil
from typing import Iterator, List
import json

from picarro.analyze import (
    ESTIMATORS,
    FluxResult,
    FluxEstimator,
    build_fluxes_dataframe,
    estimate_flux,
)
from picarro.config import AppConfig, OutItem
from picarro.measurements import (
    Measurement,
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


class ConfigProblem(Exception):
    pass


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


def identify_and_save_measurement_metas(config: AppConfig) -> None:
    path = _prepare_write_path(config, OutItem.measurement_metas_json)
    try:
        mms = list(picarro.measurements.identify_measurement_metas(config.measurements))
    except picarro.chunks.MissingColumns as e:
        raise ConfigProblem(str(e)) from e
    except picarro.chunks.InvalidData as e:
        raise U
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
    measurements = picarro.measurements.read_measurements(
        measurement_metas, config.measurements
    )
    for measurement in measurements:
        filename_stem = _build_measurement_file_name_stem(measurement)
        out_path = out_dir / f"{filename_stem}.csv"
        logger.debug(f"Writing measurement to file {out_path}.")
        measurement.to_csv(out_path)
    logger.info(f"Wrote {len(measurement_metas)} measurement(s) to files.")


def estimate_fluxes(config: AppConfig):
    if not config.flux_estimation:
        raise ConfigProblem("No flux estimation config specified.")
    path = _prepare_write_path(config, OutItem.fluxes_json)
    analysis_results = list(analyze_fluxes(config))
    columns = {ar.estimator.column for ar in analysis_results}
    n_measurements = len({ar.measurement_meta for ar in analysis_results})
    _save_analysis_results(analysis_results, path)
    logger.info(
        f"Estimated {len(analysis_results)} fluxes ({', '.join(columns)}) "
        f"in {n_measurements} measurements."
    )


def analyze_fluxes(config: AppConfig) -> Iterator[FluxResult]:
    if not config.flux_estimation:
        raise ConfigProblem("No flux estimation config specified.")
    measurement_metas = load_measurement_metas(config)
    measurements = read_measurements(measurement_metas, config.measurements)
    for measurement_meta, measurement in zip(measurement_metas, measurements):
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
        raise PreviousStepRequired("Must estimate fluxes first.")
    with open(path, "r") as f:
        obj = json.load(f)
    return _json_converter.structure(obj, List[FluxResult])


def export_fluxes_csv(config: AppConfig):
    path = _prepare_write_path(config, OutItem.fluxes_csv)

    analysis_results = _load_analysis_results(config)

    data = build_fluxes_dataframe(analysis_results)
    data.to_csv(path, index=False)
    logger.info(f"Saved results at {path}")


def _get_flux_results_by_measurement(
    config: AppConfig,
) -> dict[MeasurementMeta, List[FluxResult]]:
    result = defaultdict(list)
    for flux_result in _load_analysis_results(config):
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
    for measurement_meta, flux_results in flux_results_by_measurement.items():
        measurement = read_measurement(measurement_meta, config.measurements, cache)
        fig = picarro.plot.plot_measurement(
            measurement, config.flux_estimation.columns, flux_results
        )
        file_name = _build_measurement_file_name_stem(measurement) + ".png"
        path = out_dir / file_name
        fig.savefig(path)
        matplotlib.pyplot.close(fig)
    logger.info(f"Plotted {n_measurements} measurements.")


def _build_measurement_file_name_stem(measurement: Measurement) -> str:
    return measurement.index[0].isoformat().replace(":", "_")
