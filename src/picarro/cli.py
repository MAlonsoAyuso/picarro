from email.policy import default
from pathlib import Path
import click
import picarro.config
import picarro.app
import logging

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("picarro_config.toml")


@click.group()
@click.pass_context
@click.option(
    "--config",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    default=_DEFAULT_CONFIG_PATH,
)
def cli(ctx: click.Context, config: Path):
    ctx.ensure_object(dict)
    ctx.obj["config"] = picarro.config.AppConfig.from_toml(config)
    picarro.app.setup_logging(ctx.obj["config"])


@cli.command()
@click.pass_context
def chunks(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    mms = list(picarro.app.iter_measurement_metas(config))
    chunks = {chunk for mm in mms for chunk in mm.chunks}
    paths = {chunk.path for chunk in chunks}
    summary = (
        f"{len(mms)} measurement(s) "
        f"from {len(chunks)} chunk(s) "
        f"in {len(paths)} file(s)"
    )
    logger.info(summary)


@cli.command()
@click.pass_context
def measurements(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    picarro.app.export_measurements(config)


@cli.command()
@click.pass_context
def fluxes(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    picarro.app.export_fluxes(config)


@cli.command()
@click.pass_context
def plots(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    picarro.app.plot_fluxes(config)
