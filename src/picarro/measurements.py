from __future__ import annotations
from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, NewType, Optional, Tuple, Union
import pandas as pd
from picarro.chunks import (
    Chunk,
    ChunkMeta,
    ParsingConfig,
    read_chunks,
)
import logging

logger = logging.getLogger(__name__)


# Measurement: A DataFrame possibly constructed from one or more chunks,
#   hopefully corresponding to a whole measurement from start to end.
Measurement = NewType("Measurement", pd.DataFrame)


@dataclass(frozen=True)
class MeasurementMeta:
    chunks: Tuple[ChunkMeta, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    valve_number: int
    n_samples: int

    @staticmethod
    def from_chunk_metas(chunk_metas: List[ChunkMeta]) -> MeasurementMeta:
        valve_numbers = {c.valve_number for c in chunk_metas}
        assert len(valve_numbers) == 1, valve_numbers
        (valve_number,) = valve_numbers
        return MeasurementMeta(
            tuple(chunk_metas),
            chunk_metas[0].start,
            chunk_metas[-1].end,
            valve_number,
            sum(c.n_samples for c in chunk_metas),
        )


@dataclass
class StitchingConfig:
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_duration: Optional[pd.Timedelta] = None
    max_duration: Optional[pd.Timedelta] = None


@dataclass
class MeasurementsConfig(StitchingConfig, ParsingConfig):
    src: Union[str, List[str]] = ""


def _stitch_chunks(
    chunk_metas: Iterable[ChunkMeta], config: StitchingConfig
) -> Iterator[MeasurementMeta]:
    chunk_metas = list(chunk_metas)
    chunk_metas.sort(key=lambda c: c.start.to_numpy())

    while chunk_metas:
        collected = [chunk_metas.pop(0)]

        while chunk_metas:
            prev_chunk = collected[-1]
            candidate = chunk_metas.pop(0)

            time_gap = candidate.start - prev_chunk.end  # type: ignore
            if time_gap < pd.Timedelta(0):
                raise ValueError(f"overlapping chunks: {prev_chunk} {candidate}")

            is_adjacent = time_gap < config.max_gap
            same_valve = prev_chunk.valve_number == candidate.valve_number

            if is_adjacent and same_valve:
                collected.append(candidate)
            else:
                chunk_metas.insert(0, candidate)
                break

        yield MeasurementMeta.from_chunk_metas(collected)


def stitch_chunk_metas(
    chunk_metas: Iterable[ChunkMeta], config: StitchingConfig
) -> Iterator[MeasurementMeta]:
    measurement_metas = _stitch_chunks(chunk_metas, config)

    for measurement_meta in measurement_metas:
        duration = measurement_meta.end - measurement_meta.start  # type: ignore
        min_duration = config.min_duration
        max_duration = config.max_duration
        if (min_duration and duration < min_duration) or (
            max_duration and max_duration < duration
        ):
            logger.warning(
                f"Skipping measurement at {measurement_meta.start:%Y-%m-%d %H:%M:%S} "
                f"with duration {duration}."
            )
            logger.debug(
                f"Skipping measurement {measurement_meta}. "
                f"duration={duration}; "
                f"min_duration={min_duration}; "
                f"max_duration={max_duration}"
            )
            continue

        yield measurement_meta


def read_measurement(
    measurement_meta: MeasurementMeta,
    config: ParsingConfig,
    read_cache: Optional[Dict[ChunkMeta, Chunk]] = None,
) -> Measurement:
    if read_cache is None:
        read_cache = {}

    def read_chunks_into_cache(path: Path):
        read_cache.clear()
        read_cache.update(read_chunks(path, config))

    def read_chunk(chunk_meta: ChunkMeta):
        if chunk_meta not in read_cache:
            read_chunks_into_cache(chunk_meta.path)
        return read_cache[chunk_meta]

    measurement = pd.concat(list(map(read_chunk, measurement_meta.chunks)))
    debug_info = (measurement_meta, measurement.index)
    assert measurement_meta.start == measurement.index[0], debug_info
    assert measurement_meta.end == measurement.index[-1], debug_info
    return Measurement(measurement)


def read_measurements(
    measurement_metas: Iterable[MeasurementMeta], config: ParsingConfig
) -> Iterable[Measurement]:
    cache = {}
    for measurement_meta in measurement_metas:
        yield read_measurement(measurement_meta, config, cache)


def identify_measurement_metas(config: MeasurementsConfig) -> Iterator[MeasurementMeta]:
    chunk_metas = _iter_chunk_metas(config)
    yield from stitch_chunk_metas(chunk_metas, config)


def _iter_chunk_metas(config: MeasurementsConfig) -> Iterator[ChunkMeta]:
    glob_patterns = config.src
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    for glob_pattern in glob_patterns:
        file_paths = list(map(Path, glob.glob(glob_pattern, recursive=True)))
        logger.info(
            f"Found {len(file_paths)} source files using pattern {glob_pattern}"
        )
        for path in file_paths:
            yield from read_chunks(path, config)
