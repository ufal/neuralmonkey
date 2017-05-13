import math
from typing import Any, List
import numpy as np


class MeanSquareError(object):
    # pylint: disable=too-few-public-methods

    def __init__(self, name: str = "MeanSquareError") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[Any]],
                 references: List[List[Any]]) -> float:
        # import ipdb
        # ipdb.set_trace()
        return np.mean([np.square(float(r) -d)
                        for dec, ref in zip(decoded, references)
                        for d, r in zip(dec, ref)])

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the smaller the better
        return (score1 < score2) - (score1 > score2)
