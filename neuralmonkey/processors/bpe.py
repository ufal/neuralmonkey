import re
from typing import List

from neuralmonkey.logging import log
from lib.subword_nmt.apply_bpe import BPE, encode

# pylint: disable=too-few-public-methods


class BPEPreprocessor(object):
    """Wrapper class for Byte-Pair Encoding.

    Paper: https://arxiv.org/abs/1508.07909
    Code: https://github.com/rsennrich/subword-nmt
    """

    def __init__(self, merge_file: str, separator: str="@@") -> None:
        log("Initializing BPE preprocessor")

        with open(merge_file, "r") as f_data:
            self.bpe = BPE(f_data, separator)

    def __call__(self, sentence: List[str]) -> List[str]:
        """Adapted code from BPE.segment """

        output = []
        for word in sentence:

            # Hack. TODO: inspect why there are empty sentences
            if len(word) == 0:
                output.append(word)
                continue

            new_word = encode(word, self.bpe.bpe_codes)

            for item in new_word[:-1]:
                output.append(item + self.bpe.separator)
            output.append(new_word[-1])

        return output


class BPEPostprocessor(object):

    def __init__(self, separator: str="@@") -> None:
        esc = re.escape(separator)
        self.pattern = re.compile(esc + r" ")

    def __call__(self, decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [self.decode(s) for s in decoded_sentences]

    def decode(self, sentence: List[str]) -> List[str]:
        joined = " ".join(sentence)
        decoded = self.pattern.sub("", joined)
        splitted = decoded.split(" ")

        return splitted
