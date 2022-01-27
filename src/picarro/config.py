from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    List,
    TypeVar,
)
import toml
import cattr.preconf.tomlkit
import pandas as pd
import logging
from math import isnan

logger = logging.getLogger(__name__)

CONFIG_TIME_UNIT = "s"


_toml_converter = cattr.preconf.tomlkit.make_converter()
_toml_converter.register_structure_hook(
    pd.Timedelta, lambda v, _: pd.Timedelta(v, CONFIG_TIME_UNIT)
)


class PicarroColumns:
    DATE = "DATE"
    TIME = "TIME"
    FRAC_DAYS_SINCE_JAN1 = "FRAC_DAYS_SINCE_JAN1"
    FRAC_HRS_SINCE_JAN1 = "FRAC_HRS_SINCE_JAN1"
    JULIAN_DAYS = "JULIAN_DAYS"
    EPOCH_TIME = "EPOCH_TIME"
    ALARM_STATUS = "ALARM_STATUS"
    INST_STATUS = "INST_STATUS"
    CavityPressure = "CavityPressure"
    CavityTemp = "CavityTemp"
    DasTemp = "DasTemp"
    EtalonTemp = "EtalonTemp"
    WarmBoxTemp = "WarmBoxTemp"
    species = "species"
    MPVPosition = "MPVPosition"
    OutletValve = "OutletValve"
    solenoid_valves = "solenoid_valves"
    N2O = "N2O"
    N2O_30s = "N2O_30s"
    N2O_1min = "N2O_1min"
    N2O_5min = "N2O_5min"
    N2O_dry = "N2O_dry"
    N2O_dry30s = "N2O_dry30s"
    N2O_dry1min = "N2O_dry1min"
    N2O_dry5min = "N2O_dry5min"
    CO2 = "CO2"
    CH4 = "CH4"
    CH4_dry = "CH4_dry"
    H2O = "H2O"
    NH3 = "NH3"
    ChemDetect = "ChemDetect"
    peak_1a = "peak_1a"
    peak_41 = "peak_41"
    peak_4 = "peak_4"
    peak15 = "peak15"
    ch4_splinemax = "ch4_splinemax"
    nh3_conc_ave = "nh3_conc_ave"


Column = str


class InvalidRowHandling(Enum):
    skip = "skip"
    error = "error"


@dataclass(frozen=True)
class ParsingConfig:
    columns: List[str] = field(default_factory=list)
    null_rows: InvalidRowHandling = InvalidRowHandling.skip


@dataclass(frozen=True)
class MeasurementConfig:
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_duration: Optional[pd.Timedelta] = None
    max_duration: Optional[pd.Timedelta] = None


@dataclass(frozen=True)
class ReadConfig(ParsingConfig, MeasurementConfig):
    src: str = ""


@dataclass(frozen=True)
class FluxEstimationConfig:
    method: str
    columns: List[str]
    t0_delay: pd.Timedelta
    t0_margin: pd.Timedelta
    A: float = float("nan")
    Q: float = float("nan")
    V: float = float("nan")
    tau: float = float("nan")
    h: float = float("nan")

    def __post_init__(self):
        if isnan(self.tau):
            if isnan(self.V) or isnan(self.Q):
                raise ValueError("Must specify tau or (V, Q)")
            object.__setattr__(self, "tau", self.V / self.Q)
            logger.debug(f"calculated tau = V / Q = {self.V} / {self.Q} = {self.tau}")
        else:
            if not (isnan(self.V) or isnan(self.Q)):
                raise ValueError("Must specify tau or (V, Q), not all three")

        if isnan(self.h):
            if isnan(self.V) or isnan(self.A):
                raise ValueError("Must specify h or (V, A)")
            object.__setattr__(self, "h", self.V / self.A)
            logger.debug(f"calculated h = V / A = {self.V} / {self.A} = {self.h}")
        else:
            if not (isnan(self.V) or isnan(self.A)):
                raise ValueError("Must specify h or (V, A), not all three")

        assert not isnan(self.tau)
        assert not isnan(self.h)


@dataclass(frozen=True)
class OutputConfig:
    out_dir: Path = Path("picarro_results")


# https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema
LogSettingsDict = Dict[str, Any]

DEFAULT_LOG_SETTINGS = {
    "formatters": {
        "picarro_default": {
            "format": "%(asctime)s %(levelname)-8s %(name)-18s %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "picarro_default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "filename": "log.txt",
            "formatter": "picarro_default",
            "maxBytes": 1e6,
            "backupCount": 2,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "console",
            "file",
        ],
    },
    "version": 1,
    "disable_existing_loggers": False,
}


@dataclass(frozen=True)
class UserConfig:
    measurements: ReadConfig
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
