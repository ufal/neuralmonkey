from typing import List
from difflib import SequenceMatcher

from neuralmonkey.evaluators.evaluator import Evaluator


class EditDistanceEvaluator(Evaluator[List[str]]):

    # pylint: disable=no-self-use
    def score_instance(self,
                       hypothesis: List[str],
                       reference: List[str]) -> float:
        hyp_joined = " ".join(hypothesis)
        ref_joined = " ".join(reference)

        matcher = SequenceMatcher(None, hyp_joined, ref_joined)
        return matcher.ratio()
    # pylint: enable=no-self-use

    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:
        score = super().score_batch(hypotheses, references)
        return 1 - score

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        return super().compare_scores(score2, score1)


# pylint: disable=invalid-name
EditDistance = EditDistanceEvaluator("Edit distance")
