from typing import List, Tuple
from typeguard import check_argument_types

from neuralmonkey.evaluators.bleu import BLEUEvaluator
from neuralmonkey.evaluators.evaluator import Evaluator, check_lengths


class GLEUEvaluator(Evaluator[List[str]]):
    """Sentence-level evaluation metric correlating with BLEU on corpus-level.

    From "Google's Neural Machine Translation System: Bridging the Gap
    between Human and Machine Translation" by Wu et al.
    (https://arxiv.org/pdf/1609.08144v2.pdf)

    GLEU is the minimum of recall and precision of all n-grams up to n in
    references and hypotheses.

    Ngram counts are based on the bleu methods.
    """

    def __init__(self,
                 n: int = 4,
                 deduplicate: bool = False,
                 name: str = None) -> None:
        check_argument_types()
        if name is None:
            name = "GLEU-{}".format(n)
            if deduplicate:
                name += "-dedup"
        super().__init__(name)

        self.n = n
        self.deduplicate = deduplicate

    @check_lengths
    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:
        listed_references = [[s] for s in references]
        if self.deduplicate:
            hypotheses = BLEUEvaluator.deduplicate_sentences(hypotheses)

        return GLEUEvaluator.gleu(hypotheses, listed_references, self.n)

    # pylint: disable=too-many-locals
    @staticmethod
    def total_precision_recall(
            hypotheses: List[List[str]],
            references_list: List[List[List[str]]],
            ngrams: int,
            case_sensitive: bool) -> Tuple[float, float]:
        """Compute a modified n-gram precision and recall on a sentence list.

        Arguments:
            hypotheses: List of output sentences as lists of words
            references_list: List of lists of reference sentences (as lists of
                words)
            ngrams: n-gram order
            case_sensitive: Whether to perform case-sensitive computation
        """
        corpus_true_positives = 0
        corpus_generated_length = 0
        corpus_target_length = 0

        for n in range(1, ngrams + 1):
            for hypothesis, references in zip(hypotheses, references_list):
                reference_counters = []

                for reference in references:
                    counter = BLEUEvaluator.ngram_counts(reference, n,
                                                         not case_sensitive)
                    reference_counters.append(counter)

                reference_counts = BLEUEvaluator.merge_max_counters(
                    reference_counters)
                corpus_target_length += sum(reference_counts.values())

                hypothesis_counts = BLEUEvaluator.ngram_counts(
                    hypothesis, n, not case_sensitive)
                true_positives = 0
                for ngram in hypothesis_counts:
                    true_positives += reference_counts[ngram]

                corpus_true_positives += true_positives
                corpus_generated_length += sum(hypothesis_counts.values())

            if corpus_generated_length == 0:
                return 0, 0

        return (corpus_true_positives / corpus_generated_length,
                corpus_true_positives / corpus_target_length)

    @staticmethod
    def gleu(hypotheses: List[List[str]],
             references: List[List[List[str]]],
             ngrams: int = 4,
             case_sensitive: bool = True) -> float:
        """Compute GLEU on a corpus with multiple references (no smoothing).

        Arguments:
            hypotheses: List of hypotheses
            references: LIst of references. There can be more than one
                reference.
            ngrams: Maximum order of n-grams. Default 4.
            case_sensitive: Perform case-sensitive computation. Default True.
        """
        prec, recall = GLEUEvaluator.total_precision_recall(
            hypotheses, references, ngrams, case_sensitive)

        return min(recall, prec)
