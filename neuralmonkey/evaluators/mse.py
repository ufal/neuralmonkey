from typing import List
import numpy as np

from neuralmonkey.evaluators.evaluator import Evaluator, SequenceEvaluator


class MeanSquaredErrorEvaluator(SequenceEvaluator[float]):
    """Mean squared error evaluator.

    Assumes equal vector length across the batch (see
    `SequenceEvaluator.score_batch`)
    """

    # pylint: disable=no-self-use
    def score_token(self, hyp_elem: float, ref_elem: float) -> float:
        return (hyp_elem - ref_elem) ** 2
    # pylint: enable=no-self-use

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        return super().compare_scores(score2, score1)


class PairwiseMeanSquaredErrorEvaluator(Evaluator[List[float]]):
    """Pairwise mean squared error evaluator.

    For vectors of different dimension across the batch.
    """

    # pylint: disable=no-self-use
    def score_instance(self,
                       hypothesis: List[float],
                       reference: List[float]) -> float:
        """Compute mean square error between two vectors."""
        return np.mean([(hyp - ref) ** 2
                        for hyp, ref in zip(hypothesis, reference)])
    # pylint: enable=no-self-use

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        return super().compare_scores(score2, score1)


# pylint: disable=invalid-name
MSE = MeanSquaredErrorEvaluator()
PairwiseMSE = PairwiseMeanSquaredErrorEvaluator()
# pylint: enable=invalid-name
