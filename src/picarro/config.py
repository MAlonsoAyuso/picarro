from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import toml
import cattr.preconf.tomlkit


_toml_converter = cattr.preconf.tomlkit.make_converter()


@dataclass(frozen=True)
class ReadConfig:
    src: str
    columns: List[str]
    max_gap: int = 10
    min_length: Optional[int] = None
    max_length: Optional[int] = None


_VOLUME_UNITS = {
    "N2O": 1e-6,
    "CH4": 1e-6,
    "CO2": 1e-6,
}


@dataclass(frozen=True)
class FluxEstimationConfig:
    method: str
    t0_delay: int
    t0_margin: int
    A: float
    Q: float
    V: float
    skip_end: int = 0
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
