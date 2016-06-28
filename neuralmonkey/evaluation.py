from difflib import SequenceMatcher
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# tests: lint, mypy

BLEU_SMOOTHING = SmoothingFunction(epsilon=0.01).method1

def bleu_4(decoded, references):
    listed_references = [[s] for s in references]

    bleu_res = \
        100 * corpus_bleu(listed_references, decoded,
                          weights=[0.25, 0.25, 0.25, 0.25],
                          smoothing_function=BLEU_SMOOTHING)
    return bleu_res


def bleu_1(decoded, references):
    listed_references = [[s] for s in references]

    bleu_res = \
        100 * corpus_bleu(listed_references, decoded,
                          weights=[1.0, 0, 0, 0],
                          smoothing_function=BLEU_SMOOTHING)
    return bleu_res


def bleu_4_dedup(decoded, references):
    listed_references = [[s] for s in references]
    deduplicated_sentences = []

    for sentence in decoded:
        last_w = None
        dedup_snt = []

        for word in sentence:
            if word != last_w:
                dedup_snt.append(word)
                last_w = word

        deduplicated_sentences.append(dedup_snt)

    bleu_res = \
        100 * corpus_bleu(listed_references, deduplicated_sentences,
                          weights=[0.25, 0.25, 0.25, 0.25],
                          smoothing_function=BLEU_SMOOTHING)
    return bleu_res


def accuracy(decoded, references):
    return np.mean([d == r for dec, ref in zip(decoded, references)
                    for d, r in zip(dec, ref)])


def edit_distance(decoded, references):
    def ratio(string1, string2):
        matcher = SequenceMatcher(None, string1, string2)
        return matcher.ratio()

    return 1 - np.mean([ratio(u" ".join(ref), u" ".join(dec)) \
                        for dec, ref in zip(decoded, references)])
