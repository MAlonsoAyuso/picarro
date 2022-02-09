from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union

import cattr.preconf.tomlkit
import pandas as pd
import toml

from picarro.core import ConfigProblem
from picarro.fluxes import FluxEstimationConfig
from picarro.logging import DEFAULT_LOG_SETTINGS, LogSettingsDict
from picarro.measurements import MeasurementsConfig

logger = logging.getLogger(__name__)

CONFIG_FILE_TIME_UNIT = "s"


def structure_timedelta(v, _):
    if isinstance(v, (int, float)):
        return pd.Timedelta(v, CONFIG_FILE_TIME_UNIT)
    elif isinstance(v, str):
        hh_mm_ss = re.fullmatch(r"\d\d:\d\d:\d\d", v)
        if hh_mm_ss:
            return pd.Timedelta(v)
        mm_ss = re.fullmatch(r"\d\d:\d\d", v)
        if mm_ss:
            return pd.Timedelta(f"00:{v}")

        raise ConfigProblem(
            f"Cannot parse timedelta {v}. Use format hh:mm:ss or mm:ss."
        )


_toml_converter = cattr.preconf.tomlkit.make_converter()
_toml_converter.register_structure_hook(pd.Timedelta, structure_timedelta)
_toml_converter.register_structure_hook(Union[str, List[str]], lambda v, _: v)
_toml_converter.register_structure_hook(Path, lambda v, _: Path(v))


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

    def __post_init__(self):
        # For convenience, automatically adding any columns included in flux_estimation
        # but not included in measurements.extra_columns
        if self.flux_estimation:
            add_extra_columns = [
                c
                for c in self.flux_estimation.columns
                if c not in self.measurements.extra_columns
            ]
            self.measurements.extra_columns.extend(add_extra_columns)

    @classmethod
    def from_toml(cls, path: Path):
        with open(path, "r") as f:
            data = toml.load(f)
            if "valve_labels" in data["measurements"]:
                data["measurements"]["valve_labels"] = {
                    int(k): v for k, v in data["measurements"]["valve_labels"].items()
                }
        config = _toml_converter.structure(data, cls)
        return config
