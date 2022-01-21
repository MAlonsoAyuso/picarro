from __future__ import annotations
from dataclasses import dataclass, field
from email.mime import base
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, TypeVar
import datetime
from xml.etree.ElementTree import PI
from numpy import iterable
import toml
import cattr.preconf.tomlkit
import pandas as pd

from picarro.read import PicarroColumns

CONFIG_TIME_UNIT = "s"


_toml_converter = cattr.preconf.tomlkit.make_converter()
_toml_converter.register_structure_hook(
    pd.Timedelta, lambda v, _: pd.Timedelta(v, CONFIG_TIME_UNIT)
)

_ALWAYS_READ_COLUMNS = [
    PicarroColumns.solenoid_valves,
]


@dataclass(frozen=True)
class ReadConfig:
    src: str  # path or glob
    columns: List[str]
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_duration: Optional[pd.Timedelta] = None
    max_duration: Optional[pd.Timedelta] = None


_VOLUME_UNITS = {
    "N2O": 1e-6,
    "CH4": 1e-6,
    "CO2": 1e-6,
}


@dataclass(frozen=True)
class FluxEstimationConfig:
    method: str
    columns: List[str]
    t0_delay: pd.Timedelta
    t0_margin: pd.Timedelta
    A: float
    Q: float
    V: float
    skip_end: pd.Timedelta = pd.Timedelta(0)
    volume_prefixes: Dict[str, float] = field(default_factory=_VOLUME_UNITS.copy)


@dataclass(frozen=True)
class OutputConfig:
    out_dir: Path = Path("picarro_results")


@dataclass(frozen=True)
class UserConfig:
    measurements: ReadConfig
    flux_estimation: FluxEstimationConfig
    output: OutputConfig = field(default_factory=OutputConfig)


HT = TypeVar("HT", bound=Hashable)


def _deduplicate(src: Iterable[HT]) -> list[HT]:
    result = []
    seen = set()
    for item in src:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@dataclass
class AppPaths:
    base: Path
    cache_chunks: Path
    out: Path
    out_marker: Path
    out_measurements: Path
    out_fluxes: Path

    def cache_chunk_meta(self, data_file_path: Path) -> Path:
        assert data_file_path.is_absolute(), data_file_path
        file_name = f"{data_file_path.name}-{_repr_hash(data_file_path)}.json"
        return self.cache_chunks / file_name

    @staticmethod
    def create(base_dir: Path, out_dir: Path) -> AppPaths:
        assert base_dir.is_absolute()
        out = base_dir / out_dir
        cache = out / "cache"
        return AppPaths(
            base=base_dir,
            cache_chunks=cache / "chunks",
            out=out,
            out_marker=out / ".picarro",
            out_measurements=out / "measurements",
            out_fluxes=out / "fluxes.csv",
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

    @property
    def columns_to_read(self) -> list[str]:
        return _deduplicate(
            [
                *self.user.measurements.columns,
                *self.user.flux_estimation.columns,
                *_ALWAYS_READ_COLUMNS,
            ]
        )
