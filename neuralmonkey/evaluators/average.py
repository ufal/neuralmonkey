from typing import Any, List


# pylint: disable=too-few-public-methods
class AverageEvaluator(object):
    """Just average the numeric output of a runner."""
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, decoded: List[float], _: List[Any]) -> float:
        return sum(decoded) / len(decoded) if decoded else 0.0
# pylint: enable=too-few-public-methods
