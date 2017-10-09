from typing import List, Optional


# pylint: disable=too-few-public-methods
class ChrFEvaluator(object):
    """Compute ChrF score.

    See http://www.statmt.org/wmt15/pdf/WMT49.pdf
    """

    def __init__(self, n: int = 6, beta: float = 1,
                 ignored_symbols: Optional[List[str]] = None,
                 name: Optional[str] = None) -> None:
        self.n = n
        self.max_ord = n
        # We store the squared value of Beta
        self.beta_2 = beta**2

        if ignored_symbols is not None:
            self.ignored = ignored_symbols
        else:
            self.ignored = []

        if name is not None:
            self.name = name
        else:
            self.name = "ChrF-{}".format(beta)

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    def __call__(self, hypotheses: List[List[str]],
                 references: List[List[str]]) -> float:

        chr_f = 0.0
        length = len(hypotheses)
        for hyp, ref in zip(hypotheses, references):
            self.max_ord = self.n

            hyp_joined = " ".join(hyp)
            hyp_chars = [x for x in list(hyp_joined) if x not in self.ignored]
            hyp_ngrams = self._get_ngrams(hyp_chars, self.n)

            ref_joined = " ".join(ref)
            ref_chars = [x for x in list(ref_joined) if x not in self.ignored]
            ref_ngrams = self._get_ngrams(ref_chars, self.n)

            chr_p = 0.0
            chr_r = 0.0

            if len(hyp_chars) < 1 or len(ref_chars) < 1:
                if "".join(hyp_chars) == "".join(ref_chars):
                    chr_f += 1.0
                else:
                    chr_f += 0.0
                continue

            # ChrP
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
                    chr_p += count_matched / count_all
            chr_p = chr_p / float(self.max_ord)

            # ChrR
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
                    chr_r += count_matched / count_all
            chr_r = chr_r / float(self.max_ord)

            if chr_p != 0.0 or chr_r != 0.0:
                chr_f += ((1 + self.beta_2) * (chr_p * chr_r)
                          / ((self.beta_2 * chr_p) + chr_r))

        # Average the score over all references
        return chr_f / length

    def _get_ngrams(self, tokens, n):
        if len(tokens) < n:
            self.max_ord = len(tokens)

        ngr_dicts = []
        for m in range(1, n + 1):
            ngr_dict = {}
            for i in range(m, len(tokens)):
                ngr = "".join(tokens[i - m:i])
                ngr_dict[ngr] = ngr_dict.setdefault(ngr, 0) + 1
            ngr_dicts.append(ngr_dict)
        return ngr_dicts


# pylint: disable=invalid-name
ChrF3 = ChrFEvaluator(beta=3)
