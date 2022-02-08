from typing import Any, List, Sequence, Tuple, Type, TypeVar, overload

T = TypeVar("T")


@overload
def check_item_types(seq: Tuple[Any], item_type: Type[T]) -> Tuple[T]:
    ...


@overload
def check_item_types(seq: List[Any], item_type: Type[T]) -> List[T]:
    ...


@overload
def check_item_types(seq: Sequence[Any], item_type: Type[T]) -> Sequence[T]:
    ...


def check_item_types(seq, item_type):
    for item in seq:
        if not isinstance(item, item_type):
            raise ValueError(f"item {item!r} is not {item_type}")
    return seq
