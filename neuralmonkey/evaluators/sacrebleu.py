from typing import List
from typeguard import check_argument_types
from sacrebleu import corpus_bleu, TOKENIZERS
from neuralmonkey.evaluators.evaluator import Evaluator, check_lengths

SMOOTH_VARIANTS = ["exp", "floor", "none"]


# pylint: disable=too-few-public-methods
# TODO: Sentence-level BLEU, chrf and more could be added here.
class SacreBLEUEvaluator(Evaluator[List[str]]):
    """SacreBLEU evaluator wrapper."""

    def __init__(self,
                 name: str,
                 smooth: str = "exp",
                 smooth_floor: float = 0.0,
                 force: bool = False,
                 lowercase: bool = False,
                 tokenize: str = "none",
                 use_effective_order: bool = False) -> None:
        check_argument_types()
        super().__init__(name)

        if tokenize not in TOKENIZERS:
            raise ValueError(
                "Unknown tokenizer '{}'. You must use one of sacrebleu's "
                "tokenizers: {}".format(tokenize, str(TOKENIZERS)))

        if smooth not in SMOOTH_VARIANTS:
            raise ValueError(
                "Unknown smoothing '{}'. You must use one of sacrebleu's "
                "smoothing methods: {}".format(smooth, str(SMOOTH_VARIANTS)))

        self.smooth = smooth
        self.smooth_floor = smooth_floor
        self.force = force
        self.lowercase = lowercase
        self.tokenize = tokenize
        self.use_effective_order = use_effective_order

    @check_lengths
    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:

        hyp_joined = [" ".join(hyp) for hyp in hypotheses]
        ref_joined = [" ".join(ref) for ref in references]

        bleu = corpus_bleu(hyp_joined, [ref_joined],
                           smooth=self.smooth,
                           smooth_floor=self.smooth_floor,
                           force=self.force,
                           lowercase=self.lowercase,
                           tokenize=self.tokenize,
                           use_effective_order=self.use_effective_order)

        return bleu.score


# pylint: disable=invalid-name
SacreBLEU = SacreBLEUEvaluator("BLEU")
