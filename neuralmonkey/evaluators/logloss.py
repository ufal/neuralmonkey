import math
from typing import Any, List
import numpy as np


class KaggleLogLossEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self, name: str = "LogLoss") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[Any]],
                 references: List[List[Any]]) -> float:
        # import ipdb
        # ipdb.set_trace()
        return np.mean([-float(r) * math.log(max(1.0e-15, min(d, 1.0-1.0e-15))) - (1.0-float(r)) * math.log(1.0-max(1.0e-15, min(d, 1.0-1.0e-15)))
                        for dec, ref in zip(decoded, references)
                        for d, r in zip(dec, ref)])

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the smaller the better
        return (score1 < score2) - (score1 > score2)
