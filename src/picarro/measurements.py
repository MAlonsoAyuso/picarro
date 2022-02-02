from __future__ import annotations
from dataclasses import dataclass
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
    solenoid_valve: int
    n_samples: int

    @staticmethod
    def from_chunk_metas(chunk_metas: List[ChunkMeta]) -> MeasurementMeta:
        solenoid_valves = {c.solenoid_valve for c in chunk_metas}
        assert len(solenoid_valves) == 1, solenoid_valves
        (solenoid_valve,) = solenoid_valves
        return MeasurementMeta(
            tuple(chunk_metas),
            chunk_metas[0].start,
            chunk_metas[-1].end,
            solenoid_valve,
            sum(c.n_samples for c in chunk_metas),
        )


@dataclass
class StitchingConfig:
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_duration: Optional[pd.Timedelta] = None
    max_duration: Optional[pd.Timedelta] = None



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
            same_valve = prev_chunk.solenoid_valve == candidate.solenoid_valve

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


def read_measurements(
    measurement_metas: Iterable[MeasurementMeta],
    config: ParsingConfig,
) -> Iterator[Measurement]:
    read_cache: Dict[ChunkMeta, Chunk] = {}

    def read_chunks_into_cache(path: Path):
        read_cache.clear()
        read_cache.update(read_chunks(path, config))

    def read_chunk(chunk_meta: ChunkMeta):
        if chunk_meta not in read_cache:
            read_chunks_into_cache(chunk_meta.path)
        return read_cache[chunk_meta]

    for measurement_meta in measurement_metas:
        measurement = pd.concat(list(map(read_chunk, measurement_meta.chunks)))
        debug_info = (measurement_meta, measurement.index)
        assert measurement_meta.start == measurement.index[0], debug_info
        assert measurement_meta.end == measurement.index[-1], debug_info
        yield Measurement(measurement)
