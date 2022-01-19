from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Iterator, List
import click
import picarro.read
from picarro.read import (
    ChunkMeta,
    iter_measurements_meta,
    load_chunks_meta,
    save_measurements_meta,
)

_DEFAULT_OUT_DIR = ".picarro"

_CHUNKS_META_DIR = "chunks"
_MEASUREMENTS_META_DIR = "measurements"


@dataclass
class Settings:
    cache_dir: Path
    src_dir: Path

    def __post_init__(self):
        assert self.cache_dir.is_absolute()
        assert self.src_dir.is_absolute()


@click.group()
@click.pass_context
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_OUT_DIR,
)
def cli(ctx: click.Context, cache_dir: Path):
    ctx.ensure_object(dict)
    ctx.obj["settings"] = Settings(
        cache_dir=cache_dir.absolute(),
        src_dir=Path(".").absolute(),
    )


def _get_chunks_meta_path(settings: Settings, src_path: Path) -> Path:
    src_path = src_path.absolute().relative_to(settings.src_dir)
    return settings.cache_dir / _CHUNKS_META_DIR / (str(src_path) + ".json")


def _get_measurements_meta_path(settings: Settings, src_path: Path) -> Path:
    src_path = src_path.absolute().relative_to(settings.src_dir)
    return settings.cache_dir / _MEASUREMENTS_META_DIR / (str(src_path) + ".json")


def _create_chunks_meta(settings: Settings, data_file_path: Path):
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunks_meta_path(settings, data_file_path)
    if meta_path.exists():
        return
    data = picarro.read.read_raw(data_file_path)
    chunks_meta = picarro.read.get_chunks_metadata(data, data_file_path)
    picarro.read.save_chunks_meta(chunks_meta, meta_path)


def _load_chunks_meta(settings: Settings, data_file_path: Path) -> List[ChunkMeta]:
    assert data_file_path.is_absolute(), data_file_path
    meta_path = _get_chunks_meta_path(settings, data_file_path)
    return load_chunks_meta(meta_path)


def _iter_data_file_paths(src_path: Path) -> Iterator[Path]:
    if src_path.is_dir():
        for p in src_path.iterdir():
            yield from _iter_data_file_paths(p)
    else:
        yield src_path


@cli.command()
@click.pass_context
@click.argument("src", type=click.Path(exists=True, path_type=Path))
def map_chunks(ctx: click.Context, src: Path):
    settings = ctx.obj["settings"]
    assert isinstance(settings, Settings), settings
    for p in _iter_data_file_paths(src.absolute()):
        _create_chunks_meta(settings, p)


@cli.command()
@click.pass_context
@click.argument("src", type=click.Path(exists=True, path_type=Path))
def map_measurements(ctx: click.Context, src: Path):
    settings = ctx.obj["settings"]
    assert isinstance(settings, Settings), settings
    chunks_metas = itertools.chain(
        *(
            _load_chunks_meta(settings, path)
            for path in _iter_data_file_paths(src.absolute())
        )
    )
    save_measurements_meta(
        list(iter_measurements_meta(chunks_metas)),
        _get_measurements_meta_path(settings, src.absolute()),
    )
