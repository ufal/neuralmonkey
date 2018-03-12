#!/usr/bin/env python3.5
import unittest

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.processors.wordpiece import (
    WordpiecePreprocessor, WordpiecePostprocessor)

CORPUS = [
    "the colorless ideas slept furiously",
    "pooh slept all night",
    "working class hero is something to be",
    "I am the working class walrus",
    "walrus for president"
]

TOKENIZED_CORPUS = [[a + "_" for a in s.split()] for s in CORPUS]

# Create list of characters required to process the CORPUS with wordpieces
CORPUS_CHARS = [x for c in set("".join(CORPUS)) for x in [c, c + "_"]]
VOCABULARY = Vocabulary()

for w in CORPUS_CHARS:
    VOCABULARY.add_word(w)

for sent in TOKENIZED_CORPUS:
    VOCABULARY.add_tokenized_text(sent)


PREPROCESSOR = WordpiecePreprocessor(VOCABULARY)
POSTPROCESSOR = WordpiecePostprocessor


class TestWordpieces(unittest.TestCase):

    def test_preprocess_ok(self):
        raw = "I am the walrus".split()
        gold = "I_ am_ the_ walrus_".split()

        preprocessed = PREPROCESSOR(raw)
        self.assertSequenceEqual(preprocessed, gold)

    def test_preprocess_split(self):
        raw = "Ich bin der walrus".split()
        gold = "I c h_ b i n_ d e r_ walrus_".split()

        preprocessed = PREPROCESSOR(raw)
        self.assertSequenceEqual(preprocessed, gold)

    # TODO (#669): test encoding of UNK char/word
    def process_unk(self):
        pass

    # TODO (#669): implement wordpiece generator
    def create_wordpieces(self):
        pass


if __name__ == "__main__":
    unittest.main()
