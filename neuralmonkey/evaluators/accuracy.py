from typing import Any, List
import numpy as np


class AccuracyEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 seq_level: bool = False,
                 name: str = "Accuracy") -> None:
        self.seq_level = seq_level
        if seq_level:
            self.name = "AccuracyTop1"
        else:
            self.name = name

    def __call__(self,
                 decoded: List[List[Any]],
                 references: List[List[Any]]) -> float:
        if self.seq_level:
            collected_info = [dec == ref
                             for dec, ref in zip(decoded, references)]
        else:
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


# pylint: disable=invalid-name
Accuracy = AccuracyEvaluator()
AccuracySeqLevel = AccuracyEvaluator(seq_level=True)
# pylint: enable=invalid-name
