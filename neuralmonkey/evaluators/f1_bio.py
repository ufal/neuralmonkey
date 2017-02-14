# tests: lint, mypy

# pylint: disable=unused-import
from typing import List

# pylint: disable=invalid-name


class F1Evaluator(object):
    """ F1 evaluator for BIO tagging, e.g. NP chunking """

    def __init__(self, name="F1 measure"):
        self.name = name

    def __call__(self, decoded, references):
        # type: (List[List[str]], List[List[str]]) -> float
        assert len(decoded) == len(references)
        f1sum = 0.0
        for d, r in zip(decoded, references):
            f1sum += F1Evaluator.f1(d, r)
        return f1sum/len(decoded)

    @staticmethod
    def chunk2set(seq):
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
    def f1(decoded, reference):
        set_dec = F1Evaluator.chunk2set(decoded)
        set_ref = F1Evaluator.chunk2set(reference)

        tp = float(len(set_dec.intersection(set_ref)))
        if tp == 0.0:
            return 0.0
        precision = tp / len(set_dec)
        recall = tp / len(set_ref)
        return 2.0 * precision * recall / (precision + recall)
