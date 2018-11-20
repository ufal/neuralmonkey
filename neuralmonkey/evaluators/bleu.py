from collections import Counter
from typing import List, Tuple
import numpy as np
from typeguard import check_argument_types

from neuralmonkey.evaluators.evaluator import Evaluator


class BLEUEvaluator(Evaluator[List[str]]):

    def __init__(self, n: int = 4,
                 deduplicate: bool = False,
                 name: str = None,
                 multiple_references_separator: str = None) -> None:
        """Instantiate BLEU evaluator.

        Args:
            n: Longest n-grams considered.
            deduplicate: Flag whether repated tokes should be treated as one.
            name: Name displayed in the logs and TensorBoard.
            multiple_references_separator: Token that separates multiple
                reference sentences. If ``None``, it assumes the reference is
                one sentence only.
        """
        check_argument_types()

        if name is None:
            name = "BLEU-{}".format(n)
            if deduplicate:
                name += "-dedup"
        super().__init__(name)

        self.n = n
        self.deduplicate = deduplicate
        self.multiple_references_separator = multiple_references_separator

    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:

        if self.multiple_references_separator is None:
            listed_references = [[s] for s in references]
        else:
            listed_references = []
            for sentences in references:
                split_sentences = []
                curr_reference = []  # type: List[str]
                for tok in sentences:
                    if tok == self.multiple_references_separator:
                        split_sentences.append(curr_reference)
                        curr_reference = []
                    else:
                        curr_reference.append(tok)
                split_sentences.append(curr_reference)
                listed_references.append(split_sentences)

        if self.deduplicate:
            hypotheses = BLEUEvaluator.deduplicate_sentences(hypotheses)

        return 100 * BLEUEvaluator.bleu(hypotheses, listed_references, self.n)

    @staticmethod
    def ngram_counts(sentence: List[str], n: int,
                     lowercase: bool, delimiter: str = " ") -> Counter:
        """Get n-grams from a sentence.

        Arguments:
            sentence: Sentence as a list of words
            n: n-gram order
            lowercase: Convert ngrams to lowercase
            delimiter: delimiter to use to create counter entries
        """

        counts = Counter()  # type: Counter

        # pylint: disable=too-many-locals
        for begin in range(len(sentence) - n + 1):
            ngram = delimiter.join(sentence[begin:begin + n])
            if lowercase:
                ngram = ngram.lower()

            counts[ngram] += 1

        return counts

    @staticmethod
    def merge_max_counters(counters: List[Counter]) -> Counter:
        """Merge counters using maximum values."""
        merged = Counter()  # type: Counter

        for counter in counters:
            for key in counter:
                merged[key] = max(merged[key], counter[key])

        return merged

    @staticmethod
    def modified_ngram_precision(hypotheses: List[List[str]],
                                 references_list: List[List[List[str]]],
                                 n: int,
                                 case_sensitive: bool) -> Tuple[float, int]:
        """Compute the modified n-gram precision on a list of sentences.

        Arguments:
            hypotheses: List of output sentences as lists of words
            references_list: List of lists of reference sentences (as lists of
                words)
            n: n-gram order
            case_sensitive: Whether to perform case-sensitive computation
        """
        corpus_true_positives = 0
        corpus_generated_length = 0

        for hypothesis, references in zip(hypotheses, references_list):
            reference_counts = BLEUEvaluator.merge_max_counters([
                BLEUEvaluator.ngram_counts(ref, n, not case_sensitive)
                for ref in references])

            hypothesis_counts = BLEUEvaluator.ngram_counts(
                hypothesis, n, not case_sensitive)

            true_positives = 0
            for ngram in hypothesis_counts:
                true_positives += reference_counts[ngram]

            corpus_true_positives += true_positives
            corpus_generated_length += sum(hypothesis_counts.values())

        if corpus_generated_length == 0:
            return 1, 0

        return (corpus_true_positives / corpus_generated_length,
                corpus_generated_length)

    @staticmethod
    def effective_reference_length(
            hypotheses: List[List[str]],
            references_list: List[List[List[str]]]) -> int:
        """Compute the effective reference corpus length.

        The effective reference corpus length is based on best match length.

        Arguments:
            hypotheses: List of output sentences as lists of words
            references_list: List of lists of references (as lists of words)
        """

        eff_ref_length = 0

        for hypothesis, references in zip(hypotheses, references_list):
            hypothesis_length = len(hypothesis)

            best_diff = np.inf
            best_match_length = 0

            for reference in references:
                diff = np.abs(len(reference) - hypothesis_length)

                if diff < best_diff:
                    best_diff = diff
                    best_match_length = len(reference)

            eff_ref_length += best_match_length

        return eff_ref_length

    # pylint: disable=unused-argument
    # to mainain same API with the function above
    @staticmethod
    def minimum_reference_length(hypotheses: List[List[str]],
                                 references_list: List[List[str]]) -> int:
        """Compute the minimum reference corpus length.

        The minimum reference corpus length is based
        on the shortest reference sentence length.

        Arguments:
            hypotheses: List of output sentences as lists of words
            references_list: List of lists of references (as lists of words)
        """

        eff_ref_length = 0

        for references in references_list:
            shortest_length = np.inf

            for reference in references:
                if len(reference) < shortest_length:
                    shortest_length = len(reference)

            eff_ref_length += shortest_length

        return eff_ref_length

    @staticmethod
    def bleu(hypotheses: List[List[str]], references: List[List[List[str]]],
             ngrams: int = 4, case_sensitive: bool = True):
        """Compute BLEU on a corpus with multiple references.

        The n-grams are uniformly weighted.

        Default is to use smoothing as in reference implementation on:
        https://github.com/ufal/qtleap/blob/master/cuni_train/bin/mteval-v13a.pl#L831-L873

        Arguments:
            hypotheses: List of hypotheses
            references: LIst of references. There can be more than one
                reference.
            ngrams: Maximum order of n-grams. Default 4.
            case_sensitive: Perform case-sensitive computation. Default True.
        """
        log_bleu = 0
        weight = 1 / ngrams

        smooth = 1.0

        for order in range(1, ngrams + 1):
            prec, gen_len = BLEUEvaluator.modified_ngram_precision(
                hypotheses, references, order, case_sensitive)

            if prec == 0:
                smooth *= 2
                prec = 1 / (smooth * gen_len)

            log_bleu += weight * np.log(prec)

        # pylint: disable=invalid-name
        # the symbols 'r', 'c', and 'bp' are taken from the formula in
        # Papineni et al., it makes sense to follow the notation
        r = BLEUEvaluator.effective_reference_length(hypotheses, references)
        c = sum([len(hyp) for hyp in hypotheses])

        bp = min(1 - r / c, 0) if c != 0 else -np.inf
        log_bleu += bp

        return np.exp(log_bleu)

    @staticmethod
    def deduplicate_sentences(sentences: List[List[str]]) -> List[List[str]]:
        deduplicated_sentences = []

        for sentence in sentences:
            last_w = None
            dedup_snt = []

            for word in sentence:
                if word != last_w:
                    dedup_snt.append(word)
                    last_w = word

            deduplicated_sentences.append(dedup_snt)

        return deduplicated_sentences


# pylint: disable=invalid-name
BLEU1 = BLEUEvaluator(n=1)
BLEU4 = BLEUEvaluator(n=4)
BLEU = BLEUEvaluator()
