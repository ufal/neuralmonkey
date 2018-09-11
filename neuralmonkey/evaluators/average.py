from typing import Any, List

import numpy as np

# pylint: disable=too-few-public-methods
class AverageEvaluator:
    """Just average the numeric output of a runner."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, decoded: List[float], _: List[Any]) -> float:
        return np.mean([e for d in decoded for e in d]) if decoded else 0.0
# pylint: enable=too-few-public-methods
