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

TOKENIZED_CORPUS = [s.split(" ") for s in CORPUS]

# Create list of characters required to process the CORPUS with wordpieces
CORPUS_CHARS = [x for c in set("".join(CORPUS)) for x in [c, c + "_"]]
VOCABULARY = Vocabulary()

for w in CORPUS_CHARS:
    VOCABULARY.add_word(w)

PREPROCESSOR = WordpiecePreprocessor(VOCABULARY)
POSTPROCESSOR = WordpiecePostprocessor()


class TestWordpieces(unittest.TestCase):

    def test_vocabulary_size(self):
        self.assertTrue(len(VOCABULARY) == len(CORPUS_CHARS) + 4)

    def process(self):
        preprocessed = PREPROCESSOR(TOKENIZED_CORPUS)
        postprocessed = POSTPROCESSOR(preprocessed)

        for orig_sent, postprocessed_sent in \
                zip(TOKENIZED_CORPUS, postprocessed):
            self.assertSequenceEqual(orig_sent, postprocessed_sent)

    # TODO: test encoding of UNK char/word
    def process_unk(self):
        pass

    # TODO: implement wordpiece generator
    def create_wordpieces(self):
        pass


if __name__ == "__main__":
    unittest.main()
