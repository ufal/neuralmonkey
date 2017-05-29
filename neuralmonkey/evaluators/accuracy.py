from typing import Any, List
import numpy as np


class AccuracyEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 name: str = "Accuracy") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[List[Any]],
                 references: List[List[Any]]) -> float:
        collected_info = [d == r
                          for dec, ref in zip(decoded, references)
                          for d, r in zip(dec, ref)]
        if collected_info == []:
            mean = 0
        else:
            mean = np.mean(collected_info)
        return mean

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the bigger the better
        return (score1 > score2) - (score1 < score2)


class AccuracySeqLevelEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 name: str = "AccuracySeqLevel") -> None:
        self.name = name

    def __call__(self,
                 decoded: List[Any],
                 references: List[Any]) -> float:
        collected_info = [dec == ref
                          for dec, ref in zip(decoded, references)]

        if collected_info == []:
            mean = 0
        else:
            mean = np.mean(collected_info)
        return mean

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the bigger the better
        return (score1 > score2) - (score1 < score2)


# pylint: disable=invalid-name
Accuracy = AccuracyEvaluator()
AccuracySeqLevel = AccuracySeqLevelEvaluator()
# pylint: enable=invalid-name
