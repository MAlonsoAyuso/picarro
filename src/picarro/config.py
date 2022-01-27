from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Dict,
)
import toml
import cattr.preconf.tomlkit
import pandas as pd
import logging
from picarro.analyze import FluxEstimationConfig
from picarro.logging import DEFAULT_LOG_SETTINGS, LogSettingsDict
from picarro.read import MeasurementsConfig

logger = logging.getLogger(__name__)

CONFIG_FILE_TIME_UNIT = "s"

_toml_converter = cattr.preconf.tomlkit.make_converter()
_toml_converter.register_structure_hook(
    pd.Timedelta, lambda v, _: pd.Timedelta(v, CONFIG_FILE_TIME_UNIT)
)


@dataclass(frozen=True)
class OutputConfig:
    out_dir: Path = Path("picarro_results")


@dataclass(frozen=True)
class UserConfig:
    measurements: MeasurementsConfig
    flux_estimation: FluxEstimationConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LogSettingsDict = field(default_factory=lambda: DEFAULT_LOG_SETTINGS)


@dataclass
class AppPaths:
    base: Path
    cache_chunks: Path
    out: Path
    out_marker: Path
    out_measurements: Path
    out_fluxes: Path
    out_plot_fluxes: Path

    def cache_chunk_meta(self, data_file_path: Path) -> Path:
        assert data_file_path.is_absolute(), data_file_path
        file_name = f"{data_file_path.name}-{_repr_hash(data_file_path)}.json"
        return self.cache_chunks / file_name

    @staticmethod
    def create(base_dir: Path, out_dir: Path) -> AppPaths:
        assert base_dir.is_absolute()
        out = base_dir / out_dir
        cache = out / "cache"
        out_plot = out / "plot"
        return AppPaths(
            base=base_dir,
            cache_chunks=cache / "chunks",
            out=out,
            out_marker=out / ".picarro",
            out_measurements=out / "measurements",
            out_fluxes=out / "fluxes.csv",
            out_plot_fluxes=out_plot / "fluxes",
        )


def _repr_hash(obj: Any) -> str:
    m = sha256()
    m.update(repr(obj).encode())
    return m.hexdigest()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    user: UserConfig
    paths: AppPaths

    def __post_init__(self):
        assert self.base_dir.is_absolute()

    @staticmethod
    def from_toml(path: Path) -> AppConfig:
        base_dir = path.absolute().parent
        with open(path, "r") as f:
            data = toml.load(f)
        user_config = _toml_converter.structure(data, UserConfig)
        return AppConfig.create(base_dir, user_config)

    @staticmethod
    def create(base_dir: Path, user_config: UserConfig):
        app_paths = AppPaths.create(base_dir, user_config.output.out_dir)
        return AppConfig(base_dir, user_config, app_paths)
