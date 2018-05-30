from typing import List
import numpy as np


class MeanSquaredErrorEvaluator(object):
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 name: str = "MeanSquaredError",
                 order: int = 2,
                 rooted: bool = False) -> None:
        self.name = name
        self.order = order
        self.rooted = rooted

    def __call__(self,
                 decoded: List[List[float]],
                 references: List[List[float]]) -> float:
        func = (lambda x, y: (x - y) ** self.order)
        if self.order % 2 == 1:
            func = (lambda x, y: abs(x - y) ** self.order)
        ret = np.mean([func(d, r)
                       for dec, ref in zip(decoded, references)
                       for d, r in zip(dec, ref)])
        if self.rooted:
            ret = np.sqrt(ret)
        return ret

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the smaller the better
        return (score1 < score2) - (score1 > score2)


# pylint: disable=invalid-name
MSE = MeanSquaredErrorEvaluator(order=2, rooted=False)
RMSE = MeanSquaredErrorEvaluator(
    name="RootedMeanSquaredError", order=2, rooted=True)
MAE = MeanSquaredErrorEvaluator(
    name="MeanAverageError", order=1, rooted=False)
# pylint: enable=invalid-name
