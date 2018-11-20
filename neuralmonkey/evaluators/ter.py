from typing import List
import pyter
from neuralmonkey.evaluators.evaluator import Evaluator


# pylint: disable=too-few-public-methods
class TEREvaluator(Evaluator[List[str]]):
    """Compute TER using the pyter library."""

    # pylint: disable=no-self-use
    def score_instance(self,
                       hypothesis: List[str],
                       reference: List[str]) -> float:
        if reference and hypothesis:
            return pyter.ter(hypothesis, reference)
        if not reference and not hypothesis:
            return 0.0
        return 1.0
    # pylint: enable=no-self-use

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        return super().compare_scores(score2, score1)


TER = TEREvaluator("TER")
