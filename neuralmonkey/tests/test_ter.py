#!/usr/bin/env python3.5

import unittest

from neuralmonkey.evaluators.ter import TER
from neuralmonkey.tests.test_bleu import DECODED, REFERENCE


class TestBLEU(unittest.TestCase):

    def test_empty_decoded(self):
        self.assertEqual(TER([[] for _ in DECODED], REFERENCE), 1.0)

    def test_empty_reference(self):
        score = TER(DECODED, [[] for _ in REFERENCE])
        self.assertIsInstance(score, float)

    def test_identical(self):
        self.assertEqual(TER(REFERENCE, REFERENCE), 0.0)

    def test_empty_sentence(self):
        ref_empty = REFERENCE + [[]]
        out_empty = DECODED + [["something"]]
        score = TER(out_empty, ref_empty)
        self.assertAlmostEqual(score, .84, delta=10)

    def test_ter(self):
        score = TER(DECODED, REFERENCE)
        self.assertAlmostEqual(score, .84, delta=10)


if __name__ == "__main__":
    unittest.main()
