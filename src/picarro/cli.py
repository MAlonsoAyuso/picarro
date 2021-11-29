from pathlib import Path
import click
import picarro

_DEFAULT_OUT_DIR = ".picarro"


@click.group()
@click.pass_context
@click.option(
    "--dst-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_OUT_DIR,
)
def cli(ctx: click.Context, dst_dir: Path):
    ctx.ensure_object(dict)
    base_dir = Path(".").resolve()
    ctx.obj["base_dir"] = base_dir
    ctx.obj["dst_dir"] = base_dir / dst_dir


@cli.command()
@click.pass_context
@click.argument("src", type=click.Path(exists=True, path_type=Path))
def read(ctx: click.Context, src: Path):
    base_dir = ctx.obj["base_dir"]
    dst_dir = ctx.obj["dst_dir"]
    src_rel = src.resolve().relative_to(base_dir)
    dst_path = dst_dir / src_rel
    click.echo(f"reading from {src}")
    click.echo(dst_dir)
    d = picarro.read_raw(src)
    # d.to_csv(dst)
    click.echo(f"would write to {dst_path}")
