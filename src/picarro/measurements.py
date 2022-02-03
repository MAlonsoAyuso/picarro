from __future__ import annotations
from dataclasses import dataclass
import datetime
import glob
import itertools
from pathlib import Path
import string
from typing import Dict, Iterable, Iterator, List, NewType, Optional, Tuple, Union
import pandas as pd
from picarro.chunks import (
    Chunk,
    ChunkMeta,
    ParsingConfig,
    read_chunks,
)
import logging

from picarro.core import ConfigProblem, DataProcessingProblem

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
    valve_label: str

    @staticmethod
    def from_chunk_metas(
        chunk_metas: List[ChunkMeta], valve_labels: Optional[dict[int, str]]
    ) -> MeasurementMeta:
        valve_numbers = {c.valve_number for c in chunk_metas}
        assert len(valve_numbers) == 1, valve_numbers
        (valve_number,) = valve_numbers

        if valve_labels is None:
            valve_label = str(valve_number)
        else:
            try:
                valve_label = valve_labels[valve_number]
            except KeyError:
                raise ConfigProblem(f"No label found for valve #{valve_number}")

        return MeasurementMeta(
            tuple(chunk_metas),
            chunk_metas[0].start,
            chunk_metas[-1].end,
            valve_number,
            sum(c.n_samples for c in chunk_metas),
            valve_label,
        )


# The labels are meant to be used e.g., in file names and thus they should not contain
# punctuation such as : or /. Here is a whitelist that should be sufficient:
ALLOWED_LABEL_CHARACTERS = set(string.ascii_letters + string.digits + "_-.,*()[]{}=+*")


def _validate_label(s: str):
    violators = set(s) - ALLOWED_LABEL_CHARACTERS
    if set(s) < ALLOWED_LABEL_CHARACTERS:
        return
    raise ConfigProblem(f"Disallowed character(s) {violators!r} in label {s!r}")


@dataclass
class StitchingConfig:
    max_gap: pd.Timedelta = pd.Timedelta(10, "s")
    min_duration: Optional[pd.Timedelta] = None
    max_duration: Optional[pd.Timedelta] = None
    valve_labels: Optional[Dict[int, str]] = None

    def __post_init__(self):
        if self.valve_labels:
            for label in self.valve_labels.values():
                _validate_label(label)


@dataclass
class MeasurementsConfig(StitchingConfig, ParsingConfig):
    src: Union[str, List[str]] = ""


def identify_measurement_metas(config: MeasurementsConfig) -> Iterator[MeasurementMeta]:
    src_paths = _iter_source_paths(config.src)
    chunk_metas = itertools.chain(*(read_chunks(path, config) for path in src_paths))
    return build_measurement_metas(chunk_metas, config)


def build_measurement_metas(
    chunk_metas: Iterable[ChunkMeta], config: StitchingConfig
) -> Iterator[MeasurementMeta]:
    mms_stitched = _stitch_chunk_metas(chunk_metas, config)
    mms_filtered = _filter_measurement_metas(mms_stitched, config)
    yield from mms_filtered


def _iter_source_paths(src: Union[str, List[str]]) -> Iterator[Path]:
    glob_patterns = src
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    for glob_pattern in glob_patterns:
        file_paths = list(map(Path, glob.glob(glob_pattern, recursive=True)))
        logger.info(
            f"Found {len(file_paths)} source files using pattern {glob_pattern!r}."
        )
        yield from file_paths


def _stitch_chunk_metas(
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
                raise DataProcessingProblem(
                    f"Overlapping chunks: {prev_chunk} {candidate}"
                )

            is_adjacent = time_gap < config.max_gap
            same_valve = prev_chunk.valve_number == candidate.valve_number

            if is_adjacent and same_valve:
                collected.append(candidate)
            else:
                if same_valve:
                    logger.debug(
                        f"Not connecting chunks with a gap of {time_gap} "
                        f"starting at {prev_chunk.end}.")
                chunk_metas.insert(0, candidate) # put the candidate back in the list
                break

        yield MeasurementMeta.from_chunk_metas(collected, config.valve_labels)


def _filter_measurement_metas(
    measurement_metas: Iterable[MeasurementMeta],
    config: StitchingConfig,
) -> Iterator[MeasurementMeta]:
    n_skipped = 0
    skipped_min_duration = None
    skipped_max_duration = None
    skipped_total_duration = datetime.timedelta(0)
    total_duration = datetime.timedelta(0)
    for measurement_meta in measurement_metas:
        duration = measurement_meta.end - measurement_meta.start  # type: ignore
        total_duration += duration
        min_duration = config.min_duration
        max_duration = config.max_duration
        if (min_duration and duration < min_duration) or (
            max_duration and max_duration < duration
        ):
            n_skipped += 1
            skipped_total_duration += duration
            skipped_min_duration = (
                duration
                if not skipped_min_duration
                else min(skipped_min_duration, duration)
            )
            skipped_max_duration = (
                duration
                if not skipped_max_duration
                else max(skipped_max_duration, duration)
            )
            logger.debug(
                f"Skipping measurement at {measurement_meta.start:%Y-%m-%d %H:%M:%S} "
                f"with duration{duration}."
            )
            continue

        yield measurement_meta
    if n_skipped:
        skipped_share = skipped_total_duration / total_duration
        logger.warning(
            f"Skipped {n_skipped} measurement(s) "
            f"with a total duration of {skipped_total_duration} "
            f"({skipped_share:.1%} of the time). "
            f"Average duration of skipped measurement: "
            f"{skipped_total_duration/n_skipped}."
        )


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
