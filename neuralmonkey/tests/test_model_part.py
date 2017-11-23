#!/usr/bin/env python3.5
"""Test ModelPart class."""

import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.encoders.recurrent import SentenceEncoder


class Test(unittest.TestCase):
    """Test capabilities of model part."""
    def test_save_and_load(self):
        """Try to save and load encoder."""
        vocabulary = Vocabulary()
        vocabulary.add_word("a")
        vocabulary.add_word("b")

        checkpoint_file = tempfile.NamedTemporaryFile(delete=False)
        checkpoint_file.close()

        encoder = SentenceEncoder(
            name="enc", vocabulary=Vocabulary(), data_id="data_id",
            embedding_size=10, rnn_size=20, max_input_len=30,
            save_checkpoint=checkpoint_file.name,
            load_checkpoint=checkpoint_file.name)

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
