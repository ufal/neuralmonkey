import re
from typing import List, Callable, Pattern

import tensorflow as tf

from neuralmonkey.logging import log
from neuralmonkey.processors.helpers import pyfunc_wrapper
from lib.subword_nmt.apply_bpe import BPE, encode


def bpe_preprocess(
        merge_file: str,
        separator: str = "@@",
        encoding: str = "utf-8") -> Callable[[tf.Tensor], tf.Tensor]:
    """Get preprocess function for Byte-Pair Encoding.

    Paper: https://arxiv.org/abs/1508.07909
    Code: https://github.com/rsennrich/subword-nmt
    """
    log("Initializing BPE preprocessor")

    with open(merge_file, "r", encoding=encoding) as f_data:
        bpe = BPE(f_data, separator)

    @pyfunc_wrapper
    def preprocess(sentence: List[str]) -> List[str]:
        """Adapted code from BPE.segment."""
        output = []
        for word in sentence:

            # Hack. TODO: inspect why there are empty sentences
            if not word:
                output.append(word)
                continue

            new_word = encode(word, bpe.bpe_codes)

            for item in new_word[:-1]:
                output.append(item + bpe.separator)
            output.append(new_word[-1])

        return output

    return preprocess


def bpe_decode_sentence(sentence: List[str], pattern: Pattern) -> List[str]:
    joined = " ".join(sentence)
    decoded = pattern.sub("", joined)
    splitted = decoded.split(" ")

    return splitted


def bpe_postprocess(
        separator: str = "@@") -> Callable[[List[List[str]]], List[List[str]]]:

    esc = re.escape(separator)
    pattern = re.compile(esc + r" ")

    def decode_batch(decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [bpe_decode_sentence(s, pattern) for s in decoded_sentences]

    return decode_batch


# pylint: disable=invalid-name
def BPEPreprocessor(*args, **kwargs):
    raise NotImplementedError(
        "BPEPreprocessor class is deprecated. Use bpe_preprocess function.")


def BPEPostprocessor(*args, **kwargs):
    raise NotImplementedError(
        "BPEPostprocessor class is deprecated. Use bpe_postprocess function.")
