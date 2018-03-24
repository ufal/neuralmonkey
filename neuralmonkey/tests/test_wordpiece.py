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
ESCAPE_CHARS = "\\_u0987654321;"
C_CARON = "\\269;"
A_ACUTE = "225"


class TestWordpieces(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        vocabulary = Vocabulary()

        for c in CORPUS_CHARS + list(ESCAPE_CHARS):
            vocabulary.add_word(c)

        for sent in TOKENIZED_CORPUS:
            vocabulary.add_tokenized_text(sent)

        vocabulary.add_word(C_CARON)
        vocabulary.add_word(A_ACUTE)

        cls.preprocessor = WordpiecePreprocessor(vocabulary)
        cls.postprocessor = WordpiecePostprocessor

    def test_preprocess_ok(self):
        raw = "I am the walrus".split()
        gold = "I_ am_ the_ walrus_".split()

        preprocessed = TestWordpieces.preprocessor(raw)
        self.assertSequenceEqual(preprocessed, gold)

    def test_preprocess_split(self):
        raw = "Ich bin der walrus".split()
        gold = "I c h_ b i n_ d e r_ walrus_".split()

        preprocessed = TestWordpieces.preprocessor(raw)
        self.assertSequenceEqual(preprocessed, gold)

    def test_preprocess_unk(self):
        raw = "Ich bin der čermák".split()
        gold = "I c h_ b i n_ d e r_ \\269; e r m \\ 225 ; k_".split()

        preprocessed = TestWordpieces.preprocessor(raw)
        self.assertSequenceEqual(preprocessed, gold)

    def test_postprocess_ok(self):
        output = "I_ am_ the_ walrus_".split()
        gold = ["I am the walrus".split()]

        postprocessed = TestWordpieces.postprocessor([output])
        self.assertSequenceEqual(postprocessed, gold)

    def test_postprocess_split(self):
        output = "I c h_ b i n_ d e r_ walrus_".split()
        gold = ["Ich bin der walrus".split()]

        postprocessed = TestWordpieces.postprocessor([output])
        self.assertSequenceEqual(postprocessed, gold)

    def test_postprocess_unk(self):
        output = "I c h_ b i n_ d e r_ \\269; e r m \\ 225 ; k_".split()
        gold = ["Ich bin der čermák".split()]

        postprocessed = TestWordpieces.postprocessor([output])
        self.assertSequenceEqual(postprocessed, gold)

    # TODO (#669): implement wordpiece generator
    @unittest.skip("not implemented yet")
    def test_make_wordpieces(self):
        pass


if __name__ == "__main__":
    unittest.main()
