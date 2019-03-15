# pylint: disable=too-few-public-methods, no-self-use, unused-argument
from typing import List

import numpy as np

from neuralmonkey.evaluators.evaluator import Evaluator


class PerplexityEvaluator(Evaluator[float]):
    """Just 2 ** average the numeric output of a runner."""

    def score_batch(self,
                    hypotheses: List[float],
                    references: List[float]) -> float:
        return 2 ** np.mean(hypotheses)
