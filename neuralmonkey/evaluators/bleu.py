from collections import Counter
from typing import List, Tuple, Optional
import numpy as np


class BLEUEvaluator(object):

    def __init__(self, n: int=4, deduplicate: bool=False,
                 name: Optional[str]=None) -> None:
        self.n = n
        self.deduplicate = deduplicate

        if name is not None:
            self.name = name
        else:
            self.name = "BLEU-{}".format(n)
            if self.deduplicate:
                self.name += "-dedup"

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        listed_references = [[s] for s in references]

        if self.deduplicate:
            decoded = BLEUEvaluator.deduplicate_sentences(decoded)

        return 100 * BLEUEvaluator.bleu(decoded, listed_references, self.n)

    @staticmethod
    def ngram_counts(sentence: List[str], n: int,
                     lowercase: bool, delimiter: str=" ") -> Counter:
        """Get n-grams from a sentence

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
        """Merge counters using maximum values"""
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
        """Computes the modified n-gram precision on a list of sentences

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
            reference_counters = []

            for reference in references:
                counter = BLEUEvaluator.ngram_counts(reference, n,
                                                     not case_sensitive)
                reference_counters.append(counter)

            reference_counts = BLEUEvaluator.merge_max_counters(
                reference_counters)
            hypothesis_counts = BLEUEvaluator.ngram_counts(hypothesis, n,
                                                           not case_sensitive)

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
        """Computes the effective reference corpus length (based on best match
        length)

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
        """Computes the effective reference corpus length (based on the
        shortest reference sentence length)

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
             ngrams: int=4, case_sensitive: bool=True):
        """Computes BLEU on a corpus with multiple references using uniform
        weights. Default is to use smoothing as in reference implementation on:
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

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the bigger the better
        return (score1 > score2) - (score1 < score2)


# pylint: disable=invalid-name
BLEU1 = BLEUEvaluator(n=1)
BLEU4 = BLEUEvaluator(n=4)
BLEU = BLEUEvaluator()
