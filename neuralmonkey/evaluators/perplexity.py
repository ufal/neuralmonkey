# pylint: disable=too-few-public-methods, no-self-use, unused-argument
from typing import List

from neuralmonkey.evaluators.evaluator import Evaluator


class PerplexityEvaluator(Evaluator[float]):
    """Just 2 ** average the numeric output of a runner.

    Masked position get xent of 0. The sum of crosentropies is divided by the
    number of non-zero numbers.
    """

    def score_batch(self,
                    hypotheses: List[List[float]],
                    references: List[List[float]]) -> float:

        sum_of_all = sum(
            sum(xent for xent in hyp_xent) for hyp_xent in hypotheses)
        count_of_all = sum(
            sum(float(xent != 0.0) for xent in hyp_xent)
            for hyp_xent in hypotheses)

        if count_of_all == 0:
            return float("nan")
        return 2 ** (sum_of_all / count_of_all)
