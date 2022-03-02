import datetime
from typing import Any, List, Sequence, Tuple, Type, TypeVar, overload

T = TypeVar("T")


@overload
def ensure_tuple_of(seq: Tuple[Any], item_type: Type[T]) -> Tuple[T]:
    ...


@overload
def ensure_tuple_of(seq: List[Any], item_type: Type[T]) -> List[T]:
    ...


@overload
def ensure_tuple_of(seq: Sequence[Any], item_type: Type[T]) -> Sequence[T]:
    ...


def ensure_tuple_of(seq, item_type):
    for item in seq:
        if not isinstance(item, item_type):
            raise ValueError(f"item {item!r} is not {item_type}")
    return seq


def format_duration(duration: datetime.timedelta):
    remaining_seconds = duration.total_seconds()
    days = remaining_seconds // (3600 * 24)
    remaining_seconds -= days * (3600 * 24)
    hours = remaining_seconds // 3600
    remaining_seconds -= hours * 3600
    minutes = remaining_seconds // 60
    remaining_seconds -= minutes * 60
    days_str = f"{days:.0f} days " if days else ""
    return f"{days_str}{hours:02.0f}:{minutes:02.0f}:{remaining_seconds:02.0f}"
