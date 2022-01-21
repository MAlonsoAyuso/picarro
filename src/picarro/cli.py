from pathlib import Path
import click
import picarro.config
import picarro.app

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


@cli.command()
@click.pass_context
def export_measurements(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    picarro.app.export_measurements(config)


@cli.command()
@click.pass_context
def export_fluxes(ctx: click.Context):
    config = ctx.obj["config"]
    assert isinstance(config, picarro.config.AppConfig), config
    picarro.app.export_fluxes(config)
