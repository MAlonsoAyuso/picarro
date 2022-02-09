from __future__ import annotations

import datetime
import functools
import glob
import logging
import os
import shutil
import sqlite3
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import toml

import picarro.database
import picarro.fluxes
import picarro.logging
import picarro.plot

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("picarro_config.toml")


class PicarroPathExists(Exception):
    pass


def handle_exceptions(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PicarroPathExists as e:
            raise click.ClickException(
                f"Already exists: '{e}'. Use --force to overwrite."
            )
        # except ConfigProblem as e:
        #     raise click.ClickException(f"There is a problem with the config: {e}")
        # except DataProcessingProblem as e:
        #     raise click.ClickException(f"There was a problem processing the data: {e}")
        # except picarro.app.PreviousStepRequired as e:
        #     raise click.ClickException(
        #         f"A previous step is required before running this command: {e}"
        #     )
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
            raise click.ClickException(f"Crashed due to an unhandled exception: {e}")

    return wrapper


def add_force_option(func):
    @click.option(
        "--force/--no-force",
        default=False,
        show_default=True,
        help="Overwrite output paths.",
    )
    @functools.wraps(func)
    def wrapper(ctx, force: bool, *args, **kwargs):
        config = ctx.obj["config"]
        assert isinstance(config, AppConfig), config
        config.output.force = force
        return func(ctx, *args, **kwargs)

    return wrapper


# @click.group()
# @click.pass_context
# @click.option(
#     "config_path",
#     "--config",
#     type=click.Path(dir_okay=False, path_type=Path, exists=True),
#     default=_DEFAULT_CONFIG_PATH,
# )
# @click.option("--debug", is_flag=True, default=False)
# def cli(ctx: click.Context, config_path: Path, debug: bool):
#     config_path = config_path.resolve()
#     assert config_path.is_absolute()

#     os.chdir(config_path.parent)

#     config = picarro.config.AppConfig.from_toml(config_path)
#     if debug:
#         config.logging["root"]["level"] = "DEBUG"

#     picarro.app.setup_logging(config)

#     ctx.ensure_object(dict)
#     ctx.obj["config"] = config


# @cli.command()
# @click.pass_context
# @add_force_option
# @click.option(
#     "--identify/--no-identify",
#     default=True,
#     show_default=True,
#     help="Analyze the source files for measurements.",
# )
# @click.option(
#     "--export/--no-export",
#     default=False,
#     show_default=True,
#     help="Export the measurements as csv files.",
# )
# @handle_exceptions
# def measurements(ctx: click.Context, identify: bool, export: bool):
#     config = ctx.obj["config"]
#     assert isinstance(config, picarro.config.AppConfig), config

#     if identify:
#         picarro.app.identify_and_save_measurement_metas(config)
#         measurement_metas = picarro.app.load_measurement_metas(config)
#         logger.info(_summarize_measurements_meta(measurement_metas))

#     if export:
#         picarro.app.export_measurements(config)


# def _summarize_measurements_meta(measurement_metas: List[MeasurementMeta]) -> str:
#     chunks = {chunk for mm in measurement_metas for chunk in mm.chunks}
#     paths = {chunk.path for chunk in chunks}
#     return (
#         f"Built {len(measurement_metas)} measurements "
#         f"from {len(chunks)} chunks "
#         f"in {len(paths)} files."
#     )


# @cli.command()
# @click.pass_context
# @add_force_option
# @handle_exceptions
# def fluxes(ctx: click.Context):
#     config = ctx.obj["config"]
#     assert isinstance(config, picarro.config.AppConfig), config

#     picarro.app.estimate_and_save_fluxes(config)
#     picarro.app.export_fluxes_csv(config)


# @cli.command()
# @click.pass_context
# @add_force_option
# @click.option(
#     "--flux-fits/--no-flux-fits",
#     default=False,
#     help="Plot each measurement with the fitted functions.",
# )
# @handle_exceptions
# def plot(ctx: click.Context, flux_fits: bool):
#     config = ctx.obj["config"]
#     assert isinstance(config, picarro.config.AppConfig), config

#     if flux_fits:
#         picarro.app.plot_flux_fits(config)


@click.group()
@click.pass_context
@click.option(
    "config_path",
    "--config",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    default=_DEFAULT_CONFIG_PATH,
)
@click.option("--debug", is_flag=True, default=False)
def cli(ctx: click.Context, config_path: Path, debug: bool):
    ctx.ensure_object(dict)

    try:
        config = AppConfig.from_toml(config_path)
    except pydantic.ValidationError as e:
        raise click.ClickException(f"Incorrect config in file '{config_path}'.\n{e}")

    os.chdir(config_path.parent)

    if debug:
        config.logging["root"]["level"] = "DEBUG"

    picarro.logging.setup_logging(config.logging, config.output.outdir)

    ctx.obj["config"] = config


@cli.command("import")
@click.pass_context
def import_(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    conn = open_database(config)
    with conn:
        n_samples_total = 0
        n_files = 0
        for path in _iter_paths(config.data_import.src):
            n_samples_file = picarro.database.import_data(
                conn,
                Path(path),
                config.data_import.valve_column,
                config.data_import.columns,
            )
            n_files += 1
            n_samples_total += n_samples_file
    logger.info(f"Inserted {n_samples_total:,} samples from {n_files:,} files.")


@cli.command()
@click.pass_context
def filters(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    conn = open_database(config)
    with conn:
        picarro.database.remove_filters(conn)
        for column, filter_params in config.filters.items():
            picarro.database.apply_filter(conn, column, filter_params)
            n_excluded = picarro.database.count_excluded_samples(conn, [column])
            logger.info(f"Filter for {column!r} excludes {n_excluded:,} samples.")
        n_excluded_total = picarro.database.count_excluded_samples(
            conn, list(config.filters)
        )
        n_included_total = picarro.database.count_included_samples(conn)
        n_total = n_excluded_total + n_included_total
        share_excluded = n_excluded_total / n_total
        logger.info(
            f"Applied {len(config.filters)} filters "
            f"together excluding {n_excluded_total:,} of {n_total:,} samples "
            f"({share_excluded:.1%})."
        )


@cli.command()
@click.pass_context
def segments(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    conn = open_database(config)
    with conn:
        picarro.database.identify_segments(conn, config.segments)
    segments = list(picarro.database.iter_segments_info(conn))
    additional_info = ""
    if segments:
        median_duration = datetime.timedelta(
            seconds=round(
                np.median([segment.duration.total_seconds() for segment in segments])
            )
        )
        valves = sorted({segment.valve_number for segment in segments})
        additional_info = (
            f" with median duration {median_duration} from valves {valves}"
        )
    logger.info(f"Identified {len(segments)} segments{additional_info}.")


@cli.command()
@click.pass_context
def fluxes(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    conn = open_database(config)
    n_segments = 0
    n_estimates = 0
    with conn:
        for segment_info in picarro.database.iter_segments_info(conn):
            n_segments += 1
            segment = picarro.database.read_segment(conn, segment_info)
            for column in config.fluxes.columns:
                estimate = picarro.fluxes.estimate_flux(config.fluxes, segment[column])
                n_estimates += 1
                picarro.database.save_flux_estimate(conn, estimate)
    logger.info(
        f"Made {n_estimates} flux estimates ({', '.join(config.fluxes.columns)}) "
        f"from {n_segments} segments."
    )


@cli.command()
@click.pass_context
@add_force_option
@click.option("--flux-fits", is_flag=True, default=False)
@handle_exceptions
def plot(ctx: click.Context, flux_fits: bool):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    if flux_fits:
        _plot_flux_fits(config)


def _plot_flux_fits(config: AppConfig):
    conn = open_database(config)
    plot_dir = config.output.get_path(OutItem.flux_plots_dir)
    if plot_dir.exists():
        if not config.output.force:
            raise PicarroPathExists(plot_dir)
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True)
    with click.progressbar(
        list(picarro.database.iter_segments_info(conn)),
        label="Plotting flux fits",
        show_pos=True,
    ) as segments_infos:
        for segment_info in segments_infos:
            estimates = [
                picarro.database.read_flux_estimate(conn, segment_info.start, column)
                for column in config.fluxes.columns
            ]
            segment = picarro.database.read_segment(conn, segment_info)
            fig = picarro.plot.plot_segment(
                segment, config.fluxes.columns, estimates, config.valve_labels
            )
            plot_path = plot_dir / _build_segment_file_name(
                segment_info, ".png", config.valve_labels
            )
            fig.savefig(plot_path)
            plt.close(fig)


class DataImportConfig(pydantic.BaseModel):
    valve_column: str
    src: Union[str, List[str]] = pydantic.Field(default_factory=list)
    columns: List[str] = pydantic.Field(default_factory=list)


FiltersConfig = Dict[str, picarro.database.FilterParams]


class OutItem(Enum):
    database = auto()
    segments_summary_csv = auto()
    fluxes_csv = auto()
    flux_plots_dir = auto()


DEFAULT_PATHS = {
    OutItem.database: Path("picarro.sqlite"),
    OutItem.segments_summary_csv: Path("segments_summary.csv"),
    OutItem.fluxes_csv: Path("fluxes.csv"),
    OutItem.flux_plots_dir: Path("plots_fluxes"),
}


class OutputConfig(pydantic.BaseModel):
    outdir: Path = Path("picarro_output")
    rel_paths: Dict[OutItem, Path] = pydantic.Field(default_factory=DEFAULT_PATHS.copy)
    force: bool = False

    def get_path(self, item: OutItem) -> Path:
        return self.outdir / self.rel_paths[item]


class AppConfig(pydantic.BaseModel):
    data_import: DataImportConfig
    segments: picarro.database.SegmentingParams
    fluxes: picarro.fluxes.FluxesConfig
    filters: FiltersConfig = pydantic.Field(default_factory=dict)
    output: OutputConfig = pydantic.Field(default_factory=OutputConfig)
    valve_labels: Dict[int, str] = pydantic.Field(default_factory=dict)
    logging: picarro.logging.LogSettingsDict = pydantic.Field(
        default_factory=lambda: deepcopy(picarro.logging.DEFAULT_LOG_SETTINGS)
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # For convenience, automatically adding any columns included in
        # - flux_estimation
        # - filters
        # but not included in data_import.extra_columns
        if self.fluxes:
            self.data_import.columns.extend([
                c
                for c in self.fluxes.columns
                if c not in self.data_import.columns
            ])
        if self.filters:
            self.data_import.columns.extend([
                c
                for c in self.filters.keys()
                if c not in self.data_import.columns
            ])

    @classmethod
    def from_toml(cls, path: Path):
        with open(path, "r") as f:
            obj = toml.load(f)
        return cls.parse_obj(obj)


def open_database(config: AppConfig) -> sqlite3.Connection:
    return picarro.database.create_or_open(config.output.get_path(OutItem.database))


def _iter_paths(src: Union[str, list[str]]) -> Iterable[Path]:
    glob_patterns = [src] if isinstance(src, str) else src
    for glob_pattern in glob_patterns:
        paths = glob.glob(glob_pattern, recursive=True)
        logger.info(f"{len(paths)} files found: {glob_pattern}")
        yield from map(Path, paths)


def _build_segment_file_name(
    segment_info: picarro.database.SegmentInfo,
    suffix: str,
    valve_labels: dict[int, str],
) -> str:
    valve_label = valve_labels.get(
        segment_info.valve_number, str(segment_info.valve_number)
    )
    assert suffix == "" or suffix.startswith("."), suffix
    return f"{valve_label}-{segment_info.start:%Y%m%d-%H%M%S}{suffix}"
