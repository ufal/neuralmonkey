from typing import List, Set
from neuralmonkey.evaluators.evaluator import Evaluator


class F1Evaluator(Evaluator[List[str]]):
    """F1 evaluator for BIO tagging, e.g. NP chunking.

    The entities are annotated as beginning of the entity (B), continuation of
    the entity (I), the rest is outside the entity (O).
    """

    def score_instance(self,
                       hypothesis: List[str],
                       reference: List[str]) -> float:
        set_dec = self.chunk2set(hypothesis)
        set_ref = self.chunk2set(reference)

        true_positives = len(set_dec & set_ref)
        if true_positives == 0:
            return 0.0
        precision = true_positives / len(set_dec)
        recall = true_positives / len(set_ref)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def chunk2set(seq: List[str]) -> Set[str]:
        output = set()
        sid = ""
        inside_chunk = False
        for i, s in enumerate(seq):
            if not inside_chunk:
                if s == "B":
                    sid = str(i) + "-"
                    inside_chunk = True
            elif s != "I":
                sid += str(i - 1)
                output.add(sid)
                if s == "B":
                    sid = str(i) + "-"
                    inside_chunk = True
                else:
                    inside_chunk = False
        if inside_chunk:
            sid += str(len(seq) - 1)
            output.add(sid)
        return output


# pylint: disable=invalid-name
BIOF1Score = F1Evaluator("F1 measure")
# pylint: enable=invalid-name
