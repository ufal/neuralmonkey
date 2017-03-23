from typing import Any, List
import numpy as np


class AccuracyEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self, name: str="Accuracy") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[Any]],
                 references: List[List[Any]]) -> float:
        return np.mean([d == r
                        for dec, ref in zip(decoded, references)
                        for d, r in zip(dec, ref)])

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the bigger the better
        return (score1 > score2) - (score1 < score2)


# pylint: disable=invalid-name
Accuracy = AccuracyEvaluator()
# pylint: enable=invalid-name
