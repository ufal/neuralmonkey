from typing import List
import numpy as np


class MeanSquaredErrorEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self, name: str = "MeanSquaredError") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[float]],
                 references: List[List[float]]) -> float:
        return np.mean([(d - r) ** 2
                        for dec, ref in zip(decoded, references)
                        for d, r in zip(dec, ref)])

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the smaller the better
        return (score1 < score2) - (score1 > score2)


# pylint: disable=invalid-name
MSE = MeanSquaredErrorEvaluator()
# pylint: enable=invalid-name
