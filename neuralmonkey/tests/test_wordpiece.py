#!/usr/bin/env python3.5
import unittest

import tensorflow as tf

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.processors.wordpiece import (
    WordpiecePreprocessor, WordpiecePostprocessor)


class TestWordpieces(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        corpus = [
            "the colorless ideas slept furiously",
            "pooh slept all night",
            "working class hero is something to be",
            "I am the working class walrus",
            "walrus for president"
        ]

        tokenized_corpus = [[a + "_" for a in s.split()] for s in corpus]
        vocab_from_corpus = {w for sent in tokenized_corpus for w in sent}

        # Create list of characters required to process the CORPUS with
        # wordpieces
        corpus_chars = {x for c in set("".join(corpus)) for x in [c, c + "_"]}
        escape_chars = "\\_u0987654321;"
        c_caron = "\\269;"
        a_acute = "225"

        words = corpus_chars | set(escape_chars) | vocab_from_corpus
        vocabulary = Vocabulary(list(words) + [c_caron, a_acute])

        cls.preprocessor = WordpiecePreprocessor(vocabulary)
        cls.postprocessor = WordpiecePostprocessor

    def test_preprocess_ok(self):
        raw = "I am the walrus".split()
        gold = "I_ am_ the_ walrus_".split()

        with self.test_session():
            # pylint: disable=no-member
            # pylint decorator bug
            pre_bytes = TestWordpieces.preprocessor(raw).eval().tolist()
            # pylint: enable=no-member

            preprocessed = [tf.compat.as_text(b_word) for b_word in pre_bytes]

        self.assertSequenceEqual(preprocessed, gold)

    def test_preprocess_split(self):
        raw = "Ich bin der walrus".split()
        gold = "I c h_ b i n_ d e r_ walrus_".split()

        with self.test_session():
            # pylint: disable=no-member
            # pylint decorator bug
            pre_bytes = TestWordpieces.preprocessor(raw).eval().tolist()
            # pylint: enable=no-member

            preprocessed = [tf.compat.as_text(b_word) for b_word in pre_bytes]

        self.assertSequenceEqual(preprocessed, gold)

    def test_preprocess_unk(self):
        raw = "Ich bin der čermák".split()
        gold = "I c h_ b i n_ d e r_ \\269; e r m \\ 225 ; k_".split()

        with self.test_session():
            # pylint: disable=no-member
            # pylint decorator bug
            pre_bytes = TestWordpieces.preprocessor(raw).eval().tolist()
            # pylint: enable=no-member

            preprocessed = [tf.compat.as_text(b_word) for b_word in pre_bytes]

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
