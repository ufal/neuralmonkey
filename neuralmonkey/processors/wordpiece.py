"""Loose reimplementation of the t2t tokenizer.

Original code:
https://github.com/tensorflow/tensor2tensor/blob/v1.5.5/tensor2tensor/data_generators/tokenizer.py
"""
from typing import List
import sys
import unicodedata

from neuralmonkey.logging import log
from neuralmonkey.vocabulary import Vocabulary


ALNUM_CHAR_SET = set(
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith("L") or
        unicodedata.category(chr(i)).startswith("N")))


def wordpiece_encode(sentence: List[str], vocabulary: Vocabulary) -> List[str]:
    sent = " ".join(sentence)
    tokens_str = sent[0]
    for i in range(1, len(sent)):
        if sent[i] == " ":
            continue

        if ((sent[i] in ALNUM_CHAR_SET) != (sent[i - 1] in ALNUM_CHAR_SET)
                or sent[i - 1] == " "):
            tokens_str += " "
            tokens_str += sent[i]

    # Mark the end of each token
    # TODO (#669): escape the characters properly
    tokens = [tok + "_" for tok in tokens_str.split(" ")]

    output = []  # type: List[str]
    for tok in tokens:
        tok_start = 0
        ret = []
        while tok_start < len(tok):
            for tok_end in range(len(tok), tok_start, -1):
                subtoken = tok[tok_start:tok_end]
                if subtoken in vocabulary:
                    ret.append(subtoken)
                    tok_start = tok_end
                    break
            else:
                raise ValueError(
                    "Subword '{}' (from '{}') is not in the vocabulary"
                    .format(tok[tok_start:len(tok)], tok))
        output += ret

    return output


def wordpiece_decode(sentence: List[str]) -> List[str]:
    output = []
    for tok in sentence:
        if tok.endswith("_"):
            output.append(tok[:-1] + " ")
        else:
            output.append(tok)

    return "".join(output).rstrip(" ").split(" ")


def wordpiece_decode_batch(sentences: List[List[str]]) -> List[List[str]]:
    return [wordpiece_decode(s) for s in sentences]


def get_wordpiece_preprocessor(
        vocabulary: Vocabulary) -> Callable[[List[str]], List[str]]:
    check_argument_types()
    return lambda s: wordpiece_encode(s, vocabulary)


# pylint: disable=invalid-name
# Syntactic sugar for configuration

WordpiecePreprocessor = get_wordpiece_preprocessor
WordpiecePostprocessor = lambda: wordpiece_decode_batch

DefaultWordpiecePostprocessor = wordpiece_decode_batch
