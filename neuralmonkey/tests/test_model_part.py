#!/usr/bin/env python3.5
"""Test ModelPart class."""

import os
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal
import tensorflow as tf

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.encoders.recurrent import SentenceEncoder
from neuralmonkey.model.sequence import EmbeddedSequence


class Test(unittest.TestCase):
    """Test capabilities of model part."""

    def test_reuse(self):
        vocabulary = Vocabulary(["a", "b"])

        seq1 = EmbeddedSequence(
            name="seq1",
            vocabulary=vocabulary,
            data_id="id",
            embedding_size=10)
        seq1.register_input()

        seq2 = EmbeddedSequence(
            name="seq2",
            vocabulary=vocabulary,
            embedding_size=10,
            data_id="id")
        seq2.register_input()

        seq3 = EmbeddedSequence(
            name="seq3",
            vocabulary=vocabulary,
            data_id="id",
            embedding_size=10,
            reuse=seq1)
        seq3.register_input()

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

    def test_save_and_load(self):
        """Try to save and load encoder."""
        vocabulary = Vocabulary(["a", "b"])

        checkpoint_file = tempfile.NamedTemporaryFile(delete=False)
        checkpoint_file.close()

        encoder = SentenceEncoder(
            name="enc", vocabulary=vocabulary, data_id="data_id",
            embedding_size=10, rnn_size=20, max_input_len=30,
            save_checkpoint=checkpoint_file.name,
            load_checkpoint=checkpoint_file.name)

        encoder.input_sequence.register_input()

        # NOTE: This assert needs to be here otherwise the model has
        # no parameters since the sentence encoder is initialized lazily
        self.assertIsInstance(encoder.temporal_states, tf.Tensor)

        encoders_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="enc")

        sess_1 = tf.Session()
        sess_1.run(tf.global_variables_initializer())
        encoder.save(sess_1)

        sess_2 = tf.Session()
        sess_2.run(tf.global_variables_initializer())
        encoder.load(sess_2)

        values_in_sess_1 = sess_1.run(encoders_variables)
        values_in_sess_2 = sess_2.run(encoders_variables)

        self.assertTrue(
            all(np.all(v1 == v2) for v1, v2 in
                zip(values_in_sess_1, values_in_sess_2)))

        os.remove(checkpoint_file.name)


if __name__ == "__main__":
    unittest.main()
