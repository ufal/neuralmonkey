from typing import List, Set


class F1Evaluator(object):
    """ F1 evaluator for BIO tagging, e.g. NP chunking.

    The entities are annotated as beginning of the entity (B), continuation of
    the entity (I), the rest is outside the entity (O).
    """

    def __init__(self, name: str="F1 measure") -> None:
        self.name = name

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        assert len(decoded) == len(references)
        f1sum = 0.0
        for d, r in zip(decoded, references):
            f1sum += F1Evaluator.f1_score(d, r)
        return f1sum/len(decoded)

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
            else:
                if s != "I":
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

    @staticmethod
    def f1_score(decoded: List[str], reference: List[str]) -> float:
        set_dec = F1Evaluator.chunk2set(decoded)
        set_ref = F1Evaluator.chunk2set(reference)

        true_positives = float(len(set_dec.intersection(set_ref)))
        if true_positives == 0.0:
            return 0.0
        precision = true_positives / len(set_dec)
        recall = true_positives / len(set_ref)
        return 2.0 * precision * recall / (precision + recall)


# pylint: disable=invalid-name
BIOF1Score = F1Evaluator()
# pylint: enable=invalid-name
