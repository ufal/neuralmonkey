#!/usr/bin/env python3.5

import unittest
import tensorflow as tf
from neuralmonkey.vocabulary import Vocabulary, pad_batch


class TestVocabulary(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

        cls.corpus = [
            "the colorless ideas slept furiously",
            "pooh slept all night",
            "working class hero is something to be",
            "I am the working class walrus",
            "walrus for president"
        ]

        cls.graph = tf.Graph()

        with cls.graph.as_default():
            cls.tokenized_corpus = [s.split(" ") for s in cls.corpus]
            words = [w for sent in cls.tokenized_corpus for w in sent]
            cls.vocabulary = Vocabulary(list(set(words)))

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_all_words_in(self):
        for sentence in self.tokenized_corpus:
            for word in sentence:
                self.assertTrue(word in self.vocabulary)

    def test_unknown_word(self):
        self.assertFalse("jindrisek" in self.vocabulary)

    def test_padding(self):
        padded = pad_batch(self.tokenized_corpus)
        self.assertTrue(all(len(p) == 7 for p in padded))

    def test_weights(self):
        pass

    def test_there_and_back_self(self):

        with self.graph.as_default():
            with self.test_session() as sess:
                sess.run(tf.tables_initializer())

                padded = tf.constant(
                    pad_batch(self.tokenized_corpus, max_length=20,
                              add_start_symbol=False, add_end_symbol=True))

                vectors = tf.transpose(
                    self.vocabulary.strings_to_indices(padded))
                f_vectors = sess.run(vectors)

        sentences_again = self.vocabulary.vectors_to_sentences(f_vectors)

        for orig_sentence, reconstructed_sentence in \
                zip(self.tokenized_corpus, sentences_again):
            self.assertSequenceEqual(orig_sentence, reconstructed_sentence)


if __name__ == "__main__":
    unittest.main()
