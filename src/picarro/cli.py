from __future__ import annotations
import glob
import os
from pathlib import Path
import click
import logging

# import picarro.config
import picarro.logging
import picarro.database

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("picarro_config.toml")


# def handle_exceptions(func: Callable) -> Callable:
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except picarro.app.PicarroPathExists as e:
#             raise click.ClickException(
#                 f"Already exists: {e}. Use --force to overwrite."
#             )
#         except ConfigProblem as e:
#             raise click.ClickException(f"There is a problem with the config: {e}")
#         except DataProcessingProblem as e:
#             raise click.ClickException(f"There was a problem processing the data: {e}")
#         except picarro.app.PreviousStepRequired as e:
#             raise click.ClickException(
#                 f"A previous step is required before running this command: {e}"
#             )
#         except Exception as e:
#             logger.exception(f"Unhandled exception: {e}")
#             raise click.ClickException(f"Crashed due to an unhandled exception: {e}")

#     return wrapper


# def add_force_option(func):
#     @click.option(
#         "--force/--no-force",
#         default=False,
#         show_default=True,
#         help="Overwrite output paths.",
#     )
#     @functools.wraps(func)
#     def wrapper(ctx, force: bool, *args, **kwargs):
#         config = ctx.obj["config"]
#         assert isinstance(config, picarro.config.AppConfig), config
#         config.output.force = force
#         return func(ctx, *args, **kwargs)

#     return wrapper


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
#         f"Built {len(measurement_metas)} measurement(s) "
#         f"from {len(chunks)} chunk(s) "
#         f"in {len(paths)} file(s)."
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
@click.option("--database", type=click.Path(dir_okay=False, path_type=Path))
def cli(ctx: click.Context, config_path: Path, debug: bool, database: Path):
    ctx.ensure_object(dict)
    config_path = config_path.resolve()
    assert config_path.is_absolute()

    # ctx.obj["original_dir"] = Path.cwd()
    # os.chdir(config_path.parent)

    # config = picarro.config.AppConfig.from_toml(config_path)
    # if debug:
    #     config.logging["root"]["level"] = "DEBUG"

    picarro.logging.setup_logging(picarro.logging.DEFAULT_LOG_SETTINGS)

    # ctx.obj["config"] = config
    ctx.obj["db_path"] = database


@cli.command()
@click.pass_context
@click.option("--valve-column", type=str, required=True)
@click.option(
    "columns", "--include-column", "-c", type=str, multiple=True, required=True
)
@click.argument("paths", type=click.Path(dir_okay=False, exists=True), nargs=-1)
def import_data(
    ctx: click.Context,
    # glob_patterns: tuple[str],
    valve_column: str,
    columns: tuple[str],
    paths: tuple[Path],
):
    # config = ctx.obj["config"]
    # assert isinstance(config, picarro.config.AppConfig)
    conn = picarro.database.create_or_open(ctx.obj["db_path"])
    # for glob_pattern in glob_patterns:
    #     for path in glob.glob(glob_pattern, recursive=True):
    for path in paths:
        picarro.database.import_data(conn, path, valve_column, columns)
