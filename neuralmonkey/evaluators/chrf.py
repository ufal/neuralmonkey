from typing import List, Dict
from typeguard import check_argument_types
from neuralmonkey.evaluators.evaluator import Evaluator

# pylint: disable=invalid-name
NGramDicts = List[Dict[str, int]]
# pylint: enable=invalid-name


class ChrFEvaluator(Evaluator[List[str]]):
    """Compute ChrF score.

    See http://www.statmt.org/wmt15/pdf/WMT49.pdf
    """

    def __init__(self,
                 n: int = 6,
                 beta: float = 1.0,
                 ignored_symbols: List[str] = None,
                 name: str = None) -> None:
        check_argument_types()

        if name is None:
            name = "ChrF-{}".format(beta)
        super().__init__(name)

        self.n = n
        self.max_ord = n
        self.beta_2 = beta**2

        self.ignored = []  # type: List[str]
        if ignored_symbols is not None:
            self.ignored = ignored_symbols

    def score_instance(self,
                       hypothesis: List[str],
                       reference: List[str]) -> float:
        hyp_joined = " ".join(hypothesis)
        hyp_chars = [x for x in list(hyp_joined) if x not in self.ignored]
        hyp_ngrams = self._get_ngrams(hyp_chars, self.n)

        ref_joined = " ".join(reference)
        ref_chars = [x for x in list(ref_joined) if x not in self.ignored]
        ref_ngrams = self._get_ngrams(ref_chars, self.n)

        if not hyp_chars or not ref_chars:
            if "".join(hyp_chars) == "".join(ref_chars):
                return 1.0
            return 0.0

        precision = self.chr_p(hyp_ngrams, ref_ngrams)
        recall = self.chr_r(hyp_ngrams, ref_ngrams)

        if precision == 0.0 and recall == 0.0:
            return 0.0

        return ((1 + self.beta_2) * (precision * recall)
                / ((self.beta_2 * precision) + recall))

    def chr_r(self, hyp_ngrams: NGramDicts, ref_ngrams: NGramDicts) -> float:
        recall = 0.0
        for m in range(1, self.n + 1):
            count_all = 0
            count_matched = 0
            for ngr in ref_ngrams[m - 1]:
                ref_count = ref_ngrams[m - 1][ngr]
                count_all += ref_count
                if ngr in hyp_ngrams[m - 1]:
                    count_matched += min(ref_count, hyp_ngrams[m - 1][ngr])
            # Catch division by zero
            if count_all != 0.0:
                recall += count_matched / count_all
        return recall / float(self.max_ord)

    def chr_p(self, hyp_ngrams: NGramDicts, ref_ngrams: NGramDicts) -> float:
        precision = 0.0
        for m in range(1, self.n + 1):
            count_all = 0
            count_matched = 0
            for ngr in hyp_ngrams[m - 1]:
                hyp_count = hyp_ngrams[m - 1][ngr]
                count_all += hyp_count
                if ngr in ref_ngrams[m - 1]:
                    count_matched += min(hyp_count, ref_ngrams[m - 1][ngr])
            # Catch division by zero
            if count_all != 0.0:
                precision += count_matched / count_all

        return precision / float(self.max_ord)

    def _get_ngrams(self, tokens: List[str], n: int) -> NGramDicts:
        if len(tokens) < n:
            self.max_ord = len(tokens)

        ngr_dicts = []
        for m in range(1, n + 1):
            ngr_dict = {}  # type: Dict[str, int]
            for i in range(m, len(tokens)):
                ngr = "".join(tokens[i - m:i])
                ngr_dict[ngr] = ngr_dict.setdefault(ngr, 0) + 1
            ngr_dicts.append(ngr_dict)
        return ngr_dicts


# pylint: disable=invalid-name
ChrF3 = ChrFEvaluator(beta=3)
