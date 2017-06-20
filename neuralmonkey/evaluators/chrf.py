import pyter
from typing import List, Tuple, Optional

class ChrFEvaluator(object):
    """Compute ChrF introduced in
    http://www.statmt.org/wmt15/pdf/WMT49.pdf
    
    """

    def __init__(self, n: int = 3, beta: float = 1,
                 name: Optional[str] = None) -> None:
        self.n = n
        # We store the squared value of Beta
        self.beta_2 = beta**2

        if name is not None:
            self.name = name
        else:
            self.name = "ChrF-{}".format(n)

    def __call__(self, hypotheses: List[List[str]],
                 references: List[List[str]]) -> float:
        chr_p_all = 0
        chr_p_matched = 0
        chr_r_all = 0
        chr_r_matched = 0

        for hyp, ref in zip(hypotheses, references):
            hyp_joined = " ".join(hyp)
            hyp_chars = list(hyp_joined)
            ref_joined = " ".join(ref)
            ref_chars = list(ref_joined)

            # ChrP
            for i in range(len(hyp_chars) - self.n + 1):
                chr_p_all = chr_p_all + 1
                if "".join(hyp_chars[i:i+self.n]) in ref_joined:
                    chr_p_matched = chr_p_matched + 1

            # ChrR
            for i in range(len(ref_chars) - self.n + 1):
                chr_r_all = chr_r_all + 1
                if "".join(ref_chars[i:i+self.n]) in hyp_joined:
                    chr_r_matched = chr_r_matched + 1

        chr_p = chr_p_matched / chr_p_all
        chr_r = chr_r_matched / chr_r_all

        return (1 + self.beta_2) *\
            ((chr_p * chr_r) / (self.beta_2 * chr_p + chr_r))

ChrF3 = ChrFEvaluator(n=3)
