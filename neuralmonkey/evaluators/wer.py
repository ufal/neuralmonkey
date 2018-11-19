from typing import List
import pyter
from neuralmonkey.evaluators.evaluator import Evaluator, check_lengths


class WEREvaluator(Evaluator[List[str]]):
    """Compute WER (word error rate, used in speech recognition)."""

    # pylint: disable=no-self-use
    def score_instance(self,
                       hypothesis: List[str],
                       reference: List[str]) -> float:
        if reference and hypothesis:
            return pyter.edit_distance(hypothesis, reference)
        if not reference and not hypothesis:
            return 0.0
        return len(reference)
    # pylint: enable=no-self-use

    @check_lengths
    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:
        total_length = 0
        total_score = 0.0
        for hyp, ref in zip(hypotheses, references):
            total_score += self.score_instance(hyp, ref)
            total_length += len(ref)
        return total_score / total_length

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        return super().compare_scores(score2, score1)


WER = WEREvaluator("WER")
