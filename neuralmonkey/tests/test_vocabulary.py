#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from cStringIO import StringIO
import numpy as np

from neuralmonkey.vocabulary import Vocabulary

corpus = [
  "the colorless ideas slept furiously",
  "pooh slept all night",
  "working class hero is something to be",
  "I am the working class walrus",
  "walrus for president"
  ]

tokenized_corpus = [s.split(" ") for s in corpus]

vocabulary = Vocabulary()

for s in tokenized_corpus:
    vocabulary.add_tokenized_text(s)

class TestVacabulary(unittest.TestCase):
    def test_all_words_in(self):
        for sentence in tokenized_corpus:
            for word in sentence:
                self.assertTrue(word in vocabulary)

    def test_unknown_word(self):
        self.assertFalse("jindrisek" in vocabulary)

    def test_padding(self):
        pass

    def test_weights(self):
        pass

    def test_there_and_back_self(self):
        vectors, _ = vocabulary.sentences_to_tensor(tokenized_corpus, 20)
        senteces_again = vocabulary.vectors_to_sentences(vectors[1:])

        for orig_sentence, reconstructed_sentence in \
                zip(tokenized_corpus, senteces_again):
            self.assertSequenceEqual(orig_sentence, reconstructed_sentence)

if __name__ == "__main__":
    unittest.main()
