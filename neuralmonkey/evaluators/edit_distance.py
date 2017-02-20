from typing import List
from difflib import SequenceMatcher
import numpy as np


class EditDistanceEvaluator(object):

    def __init__(self, name: str="Edit distance") -> None:
        self.name = name

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        return 1 - np.mean([EditDistance.ratio(u" ".join(ref), u" ".join(dec))
                            for dec, ref in zip(decoded, references)])

    @staticmethod
    def ratio(str1: str, str2: str) -> float:
        matcher = SequenceMatcher(None, str1, str2)
        return matcher.ratio()

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the lower the better
        return (score1 < score2) - (score1 > score2)


# pylint: disable=invalid-name
EditDistance = EditDistanceEvaluator()
