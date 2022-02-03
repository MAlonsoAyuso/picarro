from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict,
    Optional,
    Union,
    List,
)
import toml
import cattr.preconf.tomlkit
import pandas as pd
import logging
from picarro.analyze import FluxEstimationConfig
from picarro.chunks import ParsingConfig
from picarro.logging import DEFAULT_LOG_SETTINGS, LogSettingsDict
from picarro.measurements import MeasurementsConfig, StitchingConfig

logger = logging.getLogger(__name__)

CONFIG_FILE_TIME_UNIT = "s"

_toml_converter = cattr.preconf.tomlkit.make_converter()
_toml_converter.register_structure_hook(
    pd.Timedelta, lambda v, _: pd.Timedelta(v, CONFIG_FILE_TIME_UNIT)
)
_toml_converter.register_structure_hook(Union[str, List[str]], lambda v, _: v)


class OutItem(Enum):
    measurement_metas_json = auto()
    measurements_dir = auto()
    fluxes_json = auto()
    fluxes_csv = auto()
    flux_plots_dir = auto()


DEFAULT_PATHS = {
    OutItem.measurement_metas_json: Path("measurements.json"),
    OutItem.measurements_dir: Path("measurements"),
    OutItem.fluxes_json: Path("fluxes.json"),
    OutItem.fluxes_csv: Path("fluxes.csv"),
    OutItem.flux_plots_dir: Path("plots_fluxes"),
}


@dataclass
class OutputConfig:
    out_dir: Path = Path("picarro_results")
    rel_paths: Dict[OutItem, Path] = field(default_factory=DEFAULT_PATHS.copy)
    force: bool = False

    def get_path(self, item: OutItem) -> Path:
        return self.out_dir / self.rel_paths[item]


@dataclass
class AppConfig:
    measurements: MeasurementsConfig
    flux_estimation: Optional[FluxEstimationConfig] = None
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LogSettingsDict = field(default_factory=lambda: DEFAULT_LOG_SETTINGS)

    @classmethod
    def from_toml(cls, path: Path):
        with open(path, "r") as f:
            data = toml.load(f)
        config = _toml_converter.structure(data, cls)
        return config
