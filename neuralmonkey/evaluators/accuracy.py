# tests: lint, mypy

import numpy as np


class Accuracy(object):
    # pylint: disable=too-few-public-methods

    def __init__(self, name="Accuracy"):
        self.name = name

    def __call__(self, decoded, references):
        return np.mean([d == r
                        for dec, ref in zip(decoded, references)
                        for d, r in zip(dec, ref)])

    @staticmethod
    def compare_scores(score1, score2):
        # type: (float, float) -> int
        # the bigger the better
        return (score1 > score2) - (score1 < score2)
