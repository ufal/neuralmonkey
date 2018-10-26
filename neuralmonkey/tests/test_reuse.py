#!/usr/bin/env python3.5
"""Test reusing of parameters in model part."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
import tensorflow as tf

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.model.sequence import EmbeddedSequence


class Test(unittest.TestCase):
    """Test reusing capabilities of model parts."""

    def test_reuse(self):
        vocabulary = Vocabulary()
        vocabulary.add_word("a")
        vocabulary.add_word("b")

        seq1 = EmbeddedSequence(
            name="seq1",
            vocabulary=vocabulary,
            data_id="id",
            embedding_size=10)

        seq2 = EmbeddedSequence(
            name="seq2",
            vocabulary=vocabulary,
            embedding_size=10,
            data_id="id")

        seq3 = EmbeddedSequence(
            name="seq3",
            vocabulary=vocabulary,
            data_id="id",
            embedding_size=10,
            reuse=seq1)

        # blessing
        self.assertIsNotNone(seq1.embedding_matrix)
        self.assertIsNotNone(seq2.embedding_matrix)
        self.assertIsNotNone(seq3.embedding_matrix)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        params = sess.run((seq1.embedding_matrix, seq2.embedding_matrix,
                           seq3.embedding_matrix))

        with self.assertRaises(AssertionError):
            assert_array_equal(params[0], params[1])

        assert_array_equal(params[0], params[2])



if __name__ == "__main__":
    unittest.main()
