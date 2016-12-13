#!/usr/bin/env python3

# tests: lint, mypy

import codecs
import re
from neuralmonkey.logging import log
from lib.subword_nmt.apply_bpe import BPE, encode

# pylint: disable=too-few-public-methods


class BPEPreprocessor(object):
    """ Wrapper class for Byte-Pair-Encoding from Edinburgh """

    def __init__(self, **kwargs):

        if "merge_file" not in kwargs:
            raise Exception("No merge file for BPE preprocessor")

        log("Initializing BPE preprocessor")

        separator = kwargs.get("separator", "@@")
        merge_file = kwargs["merge_file"]

        with codecs.open(merge_file, "r", "utf-8") as f_data:
            self.bpe = BPE(f_data, separator)

    def __call__(self, sentence):
        # type: (List[str]) -> List[str]
        """ Adapted code from BPE.segment """

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

    def __init__(self, **kwargs):
        self.separator = kwargs.get("separator", "@@")

        esc = re.escape(self.separator)
        self.pattern = re.compile(esc + r" ")

    def __call__(self, decoded_sentences):
        # type: (List[List[str]]) -> List[List[str]]
        return [self.decode(s) for s in decoded_sentences]

    def decode(self, sentence):
        # type: (List[str]) -> List[str]

        joined = " ".join(sentence)
        decoded = self.pattern.sub("", joined)
        splitted = decoded.split(" ")

        return splitted
