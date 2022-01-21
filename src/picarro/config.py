from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, TypeVar
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
    src: str
    columns: List[str]
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_length: Optional[pd.Timedelta] = None
    max_length: Optional[pd.Timedelta] = None


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
    cache_dir: str = ".picarro_cache"
    results_dir: str = "picarro_results"


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


@dataclass(frozen=True)
class AppConfig:
    src_dir: Path
    results_subdir: str
    user: UserConfig

    def __post_init__(self):
        assert self.src_dir.is_absolute()

    @staticmethod
    def from_toml(path: Path) -> AppConfig:
        with open(path, "r") as f:
            data = toml.load(f)
        user_config = _toml_converter.structure(data, UserConfig)
        return AppConfig(path.parent.absolute(), path.stem, user_config)

    @property
    def cache_dir_absolute(self) -> Path:
        return self.src_dir / self.user.output.cache_dir

    @property
    def results_dir_absolute(self) -> Path:
        return self.src_dir / self.user.output.results_dir

    @property
    def columns_to_read(self) -> list[str]:
        return _deduplicate(
            [
                *self.user.measurements.columns,
                *self.user.flux_estimation.columns,
                *_ALWAYS_READ_COLUMNS,
            ]
        )
