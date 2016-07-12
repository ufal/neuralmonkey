#!/usr/bin/env python3

# tests: mypy, lint

import unittest
import sys
import shutil

from neuralmonkey.evaluators.bleu_ref import BLEUReferenceImplWrapper


CORPUS_DECODED = [
    "colorful thoughts furiously sleep",
    "little piglet slept all night",
    "working working working working working be be be be be be be",
    "ich bin walrus",
    "walrus for präsident"
]

CORPUS_REFERENCE = [
    "the colorless ideas slept furiously",
    "pooh slept all night",
    "working class hero is something to be",
    "I am the working class walrus",
    "walrus for president"
]


DECODED = [d.split() for d in CORPUS_DECODED]
REFERENCE = [r.split() for r in CORPUS_REFERENCE]

LOCATION = "lib/mteval/wrap-mteval.pl"
FUNC = BLEUReferenceImplWrapper(LOCATION)

def check_perl():
    return shutil.which("perl") is not None

def check_version():
    return sys.version_info >= (3, 5)

@unittest.skipUnless(check_perl(), "Perl missing. Skipping.")
@unittest.skipUnless(check_version(), "Old Python. Skipping.")
class TestBLEU(unittest.TestCase):

    def test_empty_decoded(self):
        self.assertEqual(FUNC([[] for _ in DECODED], REFERENCE), 0)

    def test_empty_reference(self):
        score = FUNC(DECODED, [[] for _ in REFERENCE])
        self.assertIsInstance(score, float)

    def test_identical(self):
        self.assertEqual(FUNC(REFERENCE, REFERENCE), 100)

    def test_empty_sentence(self):
        ref_empty = REFERENCE + [[]]
        out_empty = DECODED + [["something"]]
        score = FUNC(out_empty, ref_empty)
        self.assertAlmostEqual(score, 15, delta=10)

    def test_bleu(self):
        score = FUNC(DECODED, REFERENCE)
        self.assertAlmostEqual(score, 15, delta=10)


if __name__ == "__main__":
    unittest.main()
