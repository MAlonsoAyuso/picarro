from __future__ import annotations

import datetime
import functools
import glob
import json
import logging
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import cattr.preconf.json
import click
import matplotlib.pyplot as plt
import pandas as pd
import pydantic
import toml

import picarro.data
import picarro.fluxes
import picarro.logging
import picarro.plot
from picarro.util import ensure_tuple_of, format_duration

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("picarro_config.toml")


class ConfigProblem(Exception):
    pass


class PicarroPathExists(Exception):
    pass


class PreviousStepRequired(Exception):
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
        except ConfigProblem as e:
            raise click.ClickException(f"There is a problem with the config: {e}")
        # except DataProcessingProblem as e:
        #     raise click.ClickException(f"There was a problem processing the data: {e}")
        except PreviousStepRequired as e:
            raise click.ClickException(
                f"A previous step is required before running this command. "
                f"The following error message was received: {e}"
            )
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


@click.group()
@handle_exceptions
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
        raise ConfigProblem(f"Incorrect config in file '{config_path}'.\n{e}")

    os.chdir(config_path.parent)

    if debug:
        config.logging["handlers"]["console"]["level"] = "DEBUG"

    picarro.logging.setup_logging(config.logging, config.output.outdir)

    ctx.obj["config"] = config


@cli.command()
@handle_exceptions
@click.pass_context
@add_force_option
def measurements(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)

    filter_summaries: list[picarro.data.FilterSummary] = []
    block_infos: list[picarro.data.BlockInfo] = []

    with click.progressbar(
        list(iter_paths(config.measurements.src)),
        label="Processing files",
        show_pos=True,
    ) as paths:
        for path in paths:
            df = picarro.data.read_picarro_file(
                path,
                config.measurements.valve_column,
                config.measurements.columns,
            )
            filter_results = picarro.data.get_data_filter_results(df, config.filters)
            filter_summaries.append(
                picarro.data.summarize_data_filter_results(filter_results)
            )
            df = picarro.data.apply_filter_results(df, filter_results)
            blocks = picarro.data.split_data_to_blocks(df, config.measurements.max_gap)
            block_infos.extend(
                [picarro.data.BlockInfo.from_block(path, block) for block in blocks]
            )
    log_data_filter_info(filter_summaries)

    all_measurement_infos = list(
        picarro.data.join_block_infos(
            block_infos,
            config.measurements.max_gap,
        )
    )
    accepted_measurement_infos = picarro.data.filter_measurements(
        all_measurement_infos,
        config.measurements.min_duration,
        config.measurements.max_duration,
    )
    log_measurement_filter_info(all_measurement_infos, accepted_measurement_infos)

    save_json(accepted_measurement_infos, config, OutItem.measurement_infos_json)


@cli.command()
@handle_exceptions
@click.pass_context
@add_force_option
def fluxes(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    try:
        measurement_infos = load_json(config, OutItem.measurement_infos_json)
    except FileNotFoundError as e:
        raise PreviousStepRequired(e)
    measurement_infos = ensure_tuple_of(measurement_infos, picarro.data.MeasurementInfo)

    estimators = []
    estimation_summaries = []
    with click.progressbar(
        measurement_infos,
        label="Processing measurements",
        show_pos=True,
    ) as measurement_infos:
        for measurement_info in measurement_infos:
            measurement = picarro.data.read_measurement(
                measurement_info,
                config.measurements.valve_column,
                config.measurements.columns,
            )
            for column in config.fluxes.gases:
                estimator = picarro.fluxes.ExponentialEstimator.fit(
                    measurement[column], config.fluxes.estimation_params
                )
                estimators.append(estimator)
                estimation_summaries.append(
                    _build_estimation_summary(measurement_info, estimator, config)
                )
    save_json(estimators, config, OutItem.flux_estimators_json)
    save_csv(pd.DataFrame(estimation_summaries), config, OutItem.fluxes_csv)


def _build_estimation_summary(
    measurement_info: picarro.data.MeasurementInfo,
    estimator: picarro.fluxes.ExponentialEstimator,
    config: AppConfig,
) -> dict:
    vol_flux = estimator.estimate_vol_flux()
    return dict(
        data_start=measurement_info.data_start,
        valve_number=measurement_info.valve_number,
        valve_label=config.valve_labels.get(measurement_info.valve_number, None),
        t0=estimator.moments.t0,
        gas=estimator.gas,
        vol_flux=vol_flux,
        molar_flux=config.fluxes.vol_flux_to_molar_flux(vol_flux),
    )


def _plot_flux_fits(config: AppConfig):
    plot_dir = config.output.get_path(OutItem.flux_plots_dir)
    if plot_dir.exists():
        if not config.output.force:
            raise PicarroPathExists(plot_dir)
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True)

    try:
        measurement_infos = measurement_infos = ensure_tuple_of(
            load_json(config, OutItem.measurement_infos_json),
            picarro.data.MeasurementInfo,
        )
    except FileNotFoundError as e:
        raise PreviousStepRequired(e)

    all_estimators = ensure_tuple_of(
        load_json(config, OutItem.flux_estimators_json),
        picarro.fluxes.ExponentialEstimator,
    )
    estimators_by_start_time = defaultdict(list)
    for estimator in all_estimators:
        estimators_by_start_time[estimator.moments.data_start].append(estimator)

    with click.progressbar(
        measurement_infos,
        label="Plotting measurements",
        show_pos=True,
    ) as measurement_infos:
        for measurement_info in measurement_infos:
            assert isinstance(measurement_info, picarro.data.MeasurementInfo)
            measurement = picarro.data.read_measurement(
                measurement_info,
                config.measurements.valve_column,
                config.measurements.columns,
            )
            fig = picarro.plot.plot_measurement(
                measurement,
                config.fluxes.gases,
                estimators_by_start_time[measurement_info.data_start],
                config.valve_labels,
            )
            plot_path = plot_dir / _build_measurement_file_name(
                measurement_info, ".png", config.valve_labels
            )
            fig.savefig(plot_path)
            plt.close(fig)


_PLOT_FUNCTIONS = {
    "flux-fits": _plot_flux_fits,
}


@cli.command()
@handle_exceptions
@click.pass_context
@add_force_option
@click.argument("plot_name", type=click.Choice(list(_PLOT_FUNCTIONS)))
def plot(ctx: click.Context, plot_name: str):
    config = ctx.obj["config"]
    assert isinstance(config, AppConfig)
    _PLOT_FUNCTIONS[plot_name](config)


class MeasurementConfig(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    valve_column: str
    src: Union[str, List[str]] = pydantic.Field(default_factory=list)
    columns: List[str] = pydantic.Field(default_factory=list)
    max_gap: datetime.timedelta = datetime.timedelta(seconds=10)
    min_duration: Optional[datetime.timedelta] = datetime.timedelta(seconds=60)
    max_duration: Optional[datetime.timedelta] = None


FiltersConfig = Dict[str, picarro.data.FilterParams]


class FluxesConfig(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    t0_delay: datetime.timedelta
    t0_margin: datetime.timedelta
    A: float
    V: float
    Q: float  # the units for Q and V must be such that V/Q has unit second.
    R: float = 8.31447  # ideal gas constant in SI units
    T: Optional[float] = None
    P: Optional[float] = None

    gases: Tuple[str, ...] = ()

    @property
    def estimation_params(self) -> picarro.fluxes.EstimationParams:
        return picarro.fluxes.EstimationParams(
            self.t0_delay,
            self.t0_margin,
            self.tau,
            self.h,
        )

    @property
    def tau(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.V / self.Q)

    @property
    def h(self) -> float:
        return self.V / self.A

    def vol_flux_to_molar_flux(self, vol_flux: float) -> Optional[float]:
        if self.T and self.P:
            return vol_flux * self.P / (self.R * self.T)  # ideal gas law PV = nRT
            # If data and parameters would be in SI units, this would be too.
            # But in practice it is probably not, since our PICARRO data files
            # give concentrations in ppm.
            # Therefore the units of the mass flux is in MICROmoles/m2/s.


class OutItem(Enum):
    measurement_infos_json = auto()
    flux_estimators_json = auto()
    fluxes_csv = auto()
    flux_plots_dir = auto()


DEFAULT_PATHS = {
    OutItem.measurement_infos_json: Path("measurements.json"),
    OutItem.flux_estimators_json: Path("flux_estimators.json"),
    OutItem.fluxes_csv: Path("fluxes.csv"),
    OutItem.flux_plots_dir: Path("plots_fluxes"),
}

assert set(DEFAULT_PATHS) == set(OutItem)

DEFAULT_OUTDIR = Path("picarro_output")


class OutputConfig(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    outdir: Path = DEFAULT_OUTDIR
    rel_paths: Dict[OutItem, Path] = pydantic.Field(default_factory=DEFAULT_PATHS.copy)
    force: bool = False

    def get_path(self, item: OutItem) -> Path:
        return self.outdir / self.rel_paths[item]


class AppConfig(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    filters: FiltersConfig = pydantic.Field(default_factory=dict)
    measurements: MeasurementConfig
    fluxes: FluxesConfig
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
        # but not explicitily listed for reading
        if self.fluxes:
            self.measurements.columns.extend(
                [c for c in self.fluxes.gases if c not in self.measurements.columns]
            )
        if self.filters:
            self.measurements.columns.extend(
                [c for c in self.filters.keys() if c not in self.measurements.columns]
            )

    @classmethod
    def from_toml(cls, config_path: Path):
        with open(config_path, "r") as f:
            obj = toml.load(f)
        output_conf = obj.setdefault("output", {})
        output_conf.setdefault(
            "outdir",
            Path(output_conf.get("outdir", DEFAULT_OUTDIR)) / config_path.stem,
        )
        return cls.parse_obj(obj)


_json_converter = cattr.preconf.json.make_converter()
_json_converter.register_unstructure_hook(Path, str)
_json_converter.register_structure_hook(Path, lambda v, _: Path(v))

_json_converter.register_unstructure_hook(
    datetime.datetime, datetime.datetime.isoformat
)
_json_converter.register_unstructure_hook(
    datetime.timedelta, datetime.timedelta.total_seconds
)
_json_converter.register_structure_hook(
    datetime.datetime, lambda v, _: datetime.datetime.fromisoformat(v)
)
_json_converter.register_structure_hook(
    datetime.timedelta, lambda v, _: datetime.timedelta(seconds=v)
)


_OUT_ITEM_TYPES = {
    OutItem.measurement_infos_json: Tuple[picarro.data.MeasurementInfo, ...],
    OutItem.flux_estimators_json: Tuple[picarro.fluxes.ExponentialEstimator, ...],
}


def save_json(obj: Any, config: AppConfig, out_item: OutItem):
    path = config.output.get_path(out_item)
    json_obj = _json_converter.unstructure(
        obj, unstructure_as=_OUT_ITEM_TYPES[out_item]
    )
    mode = "w" if config.output.force else "x"
    with open(path, mode) as f:
        json.dump(json_obj, f)


def load_json(config: AppConfig, out_item: OutItem) -> Any:
    path = config.output.get_path(out_item)
    with open(path, "r") as f:
        json_obj = json.load(f)
    obj = _json_converter.structure(json_obj, _OUT_ITEM_TYPES[out_item])
    return obj


def save_csv(df: pd.DataFrame, config: AppConfig, out_item: OutItem):
    # Index is not written to file. To prevent accidental loss of any meaningful
    # data in the index, check that it is just a RangeIndex.
    # To write a DataFrame including index to csv, do df.reset_index() or similar.
    assert isinstance(df.index, pd.RangeIndex), df.index
    path = config.output.get_path(out_item)
    mode = "w" if config.output.force else "x"
    with open(path, mode) as f:
        df.to_csv(f, index=False)


def _build_measurement_file_name(
    measurement_info: picarro.data.MeasurementInfo,
    suffix: str,
    valve_labels: dict[int, str],
) -> str:
    valve_label = valve_labels.get(
        measurement_info.valve_number, str(measurement_info.valve_number)
    )
    assert suffix == "" or suffix.startswith("."), suffix
    return f"{valve_label}-{measurement_info.data_start:%Y%m%d-%H%M%S}{suffix}"


def iter_paths(src: Union[str, list[str]]) -> Iterator[Path]:
    glob_patterns = src
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    for glob_pattern in glob_patterns:
        file_paths = list(map(Path, glob.glob(glob_pattern, recursive=True)))
        logger.info(
            f"Found {len(file_paths)} source files using pattern {glob_pattern!r}."
        )
        yield from file_paths


def log_data_filter_info(filter_summaries: Iterable[picarro.data.FilterSummary]):
    filter_summaries = list(filter_summaries)
    n_files = len(filter_summaries)
    n_rows = sum(fs.n_rows for fs in filter_summaries)
    n_removed_total = sum(fs.n_removed_total for fs in filter_summaries)
    n_removed_by_col = pd.DataFrame(
        [fs.n_removed_by_col for fs in filter_summaries]
    ).sum()

    lines = []
    lines.append(
        f"Filtered {n_files} files, rejecting {n_removed_total:,} of {n_rows:,} rows "
        f"({n_removed_total / n_rows:.1%} of total)."
    )
    for col, n_removed in n_removed_by_col.items():
        assert isinstance(n_removed, int), n_removed
        lines.append(
            f"Filter '{col}': rejected {n_removed:,} rows "
            f"({n_removed / n_rows:.1%} of total)."
        )

    for line in lines:
        logger.info(line)


def log_measurement_filter_info(
    all_infos: list[picarro.data.MeasurementInfo],
    accepted: list[picarro.data.MeasurementInfo],
):
    skipped = set(all_infos) - set(accepted)
    duration_skipped = _sum_durations(skipped)
    duration_accepted = _sum_durations(accepted)
    duration_total = _sum_durations(all_infos)
    logger.info(
        f"Identified {len(all_infos)} measurements "
        f"with a total duration of {format_duration(duration_total)}. "
    )
    logger.info(
        f"Skipped {len(skipped)} measurements "
        f"with a total duration of {format_duration(duration_skipped)} "
        f"({duration_skipped/duration_total:.1%} of the time). "
        f"Average duration of skipped measurement: "
        f"{format_duration(duration_skipped/len(skipped))}."
    )
    logger.info(
        f"Accepted {len(accepted)} measurements "
        f"with a total duration of {format_duration(duration_accepted)}. "
        f"Average duration of accepted measurement: "
        f"{format_duration(duration_accepted/len(accepted))}."
    )


def _sum_durations(
    measurements_infos: Iterable[picarro.data.MeasurementInfo],
) -> datetime.timedelta:
    total = datetime.timedelta(0)
    for m in measurements_infos:
        total += m.duration
    return total
