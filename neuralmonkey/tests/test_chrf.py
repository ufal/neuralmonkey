#!/usr/bin/env python3.5


import unittest

from neuralmonkey.evaluators.chrf import ChrFEvaluator, _get_ngrams
from neuralmonkey.tests.test_bleu import DECODED, REFERENCE


TOKENS = ["a", "b", "a"]
NGRAMS = [
    {"a": 2, "b": 1},
    {"ab": 1, "ba": 1},
    {"aba": 1},
    {}]

FUNC = ChrFEvaluator()
FUNC_P = FUNC.chr_p
FUNC_R = FUNC.chr_r


class TestChrF(unittest.TestCase):

    def test_empty_decoded(self):
        # Recall == 0.0
        self.assertEqual(FUNC([[] for _ in DECODED], REFERENCE), 0.0)

    def test_empty_reference(self):
        # Precision == 0.0
        self.assertEqual(FUNC([[] for _ in REFERENCE], DECODED), 0.0)

    def test_identical(self):
        self.assertEqual(FUNC(REFERENCE, REFERENCE), 1.0)

    def test_empty_sentence(self):
        ref_empty = REFERENCE + [[]]
        out_empty = DECODED + [["something"]]
        score = FUNC(out_empty, ref_empty)
        self.assertAlmostEqual(score, 0.38, delta=10)

    def test_chrf(self):
        score = FUNC(DECODED, REFERENCE)
        self.assertAlmostEqual(score, 0.46, delta=10)

    def test_get_ngrams(self):
        tokens = ["a", "b", "a"]
        ngrams_out = _get_ngrams(tokens, 4)
        self.assertEqual(len(ngrams_out), 4)
        for i, _ in enumerate(NGRAMS):
            self.assertDictEqual(ngrams_out[i], NGRAMS[i])


if __name__ == "__main__":
    unittest.main()
