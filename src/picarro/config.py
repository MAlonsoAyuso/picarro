from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import toml
import cattr
import cattr.preconf.tomlkit


_toml_converter = cattr.preconf.tomlkit.make_converter()


@dataclass
class ReadConfig:
    src: str
    columns: List[str]
    max_gap: int = 10
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass
class FitConfig:
    method: str
    t0: int
    t0_margin: int
    A: float
    Q: float
    V: float


@dataclass
class OutputConfig:
    cache_dir: str = ".picarro_cache"
    results_dir: str = "picarro_results"


@dataclass
class UserConfig:
    measurements: ReadConfig
    fit: FitConfig
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class AppConfig:
    src_dir: Path
    results_subdir: str
    user: UserConfig

    @staticmethod
    def from_toml(path: Path) -> AppConfig:
        with open(path, "r") as f:
            data = toml.load(f)
        user_config = _toml_converter.structure(data, UserConfig)
        return AppConfig(path.parent, path.stem, user_config)
