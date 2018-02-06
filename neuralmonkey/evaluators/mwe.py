from typing import Dict, List, Tuple

import itertools


# pylint: disable=too-few-public-methods
class MWEEvaluator(object):
    """Compute the MWE counts for measuring precision, recall and F1.

    The code is based on TODO.
    """

    def __init__(self,
                 name: str = "mwe_evaluator",
                 mode: str = "mwe_based",
                 metric: str = "f-measure") -> None:
        self.name = name
        self.mode = mode
        self.metric = metric
        self.total_hyp = 0
        self.total_ref = 0
        self.correct = 0

    def __call__(self, hypotheses: List[List[str]],
                 references: List[List[str]]) -> float:
        self.total_hyp = 0
        self.total_ref = 0
        self.correct = 0
        for hyp, ref in zip(hypotheses, references):
            h_mwes = _to_mwes(hyp)
            r_mwes = _to_mwes(ref)

            if self.mode == "mwe_based":
                pairing = {x: x for x in set(h_mwes) & set(r_mwes)}
                self._increment(len(h_mwes), len(r_mwes), len(pairing))
            elif self.mode == "tok_based":
                pairing = _tok_based_pairing(h_mwes, r_mwes)
                self._increment(
                    sum(len(m) for m in h_mwes if m),
                    sum(len(m) for m in r_mwes if m),
                    sum(len(a & b) for (a, b) in pairing.items()))

        precision = self.correct / (self.total_hyp or 1.0)
        recall = self.correct / (self.total_ref or 1.0)
        f_measure = 0.0
        if precision > 0.0:
            f_measure = 2.0 * precision * recall / (precision + recall)

        ret = None
        if self.metric == "precision":
            ret = precision
        elif self.metric == "recall":
            ret = recall
        elif self.metric == "f-measure":
            ret = f_measure
        else:
            ValueError("Unknown metric name: {}".format(self.metric))

        return ret

    def _increment(self, plus_h, plus_r, plus_correct):
        self.total_hyp += plus_h
        self.total_ref += plus_r
        self.correct += plus_correct


def _to_mwes(sent) -> List[int]:
    mwe_infos = {}  # type: Dict[int, Tuple[str, List[int]]]
    for tok_idx, tok in enumerate(sent):
        # parse the token to the MWE codes
        mwe_codes = [] if tok == "_" else tok.split(";")
        for mwe_code in mwe_codes:
            # Format: "<int><:Optional[str]>"
            split = mwe_code.split(":")

            if len(split) == 1:
                split.append(None)
            mwe_info = mwe_infos.setdefault(int(split[0]), (split[1], []))
            mwe_info[1].append(tok_idx)

    mwe_set = set(frozenset(i + 1 for i in x[1]) for x in mwe_infos.values())
    return sorted(mwe_set, key=lambda x: list(x))


def _tok_based_pairing(h_mwes, r_mwes):
    """Look for the largest possible pairing.

    The simplest straightforward O(n!) algorithm.
    """
    h_mwes += [None] * (len(r_mwes) - len(h_mwes))
    r_mwes += [None] * (len(h_mwes) - len(r_mwes))
    ret, ret_count = {}, 0
    for h_mwes_permut in itertools.permutations(h_mwes):
        pairing = {a: b for (a, b) in zip(r_mwes, h_mwes_permut) if a and b}
        pairing_count = sum(
            len(set(a) & set(b)) for (a, b) in pairing.items())
        if pairing_count > ret_count:
            ret, ret_count = pairing, pairing_count
    return ret
