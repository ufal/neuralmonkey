from typing import TypeVar, Callable, List
from typeguard import check_argument_types

T = TypeVar("T")


def strip(left: int, right: int) -> Callable[[List[T]], List[T]]:
    check_argument_types()

    if left < 0 or right < 0:
        raise ValueError("Sequence stripping only supports positive numbers")

    def process(sequence: List[T]) -> List[T]:
        return sequence[left:len(sequence) - right]

    return process


def left_strip(positions: int) -> Callable[[List[T]], List[T]]:
    return strip(left=positions, right=0)


def right_strip(positions: int) -> Callable[[List[T]], List[T]]:
    return strip(left=0, right=positions)
