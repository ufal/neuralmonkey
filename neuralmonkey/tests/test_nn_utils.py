#!/usr/bin/env python3.5

import unittest

import numpy as np
import tensorflow as tf
from neuralmonkey.nn.utils import dropout


class TestDropout(unittest.TestCase):

    def test_invalid_keep_prob(self):
        """Tests invalid dropout values"""

        var = tf.constant(np.arange(5))
        train_mode = tf.constant(True)

        for kprob in [-1, 2, 0]:
            with self.assertRaises(ValueError):
                dropout(var, kprob, train_mode)

    def test_keep_prob(self):
        """Counts dropped items and compare with the expectation"""

        var = tf.ones([10000])
        s = tf.Session()

        for kprob in [0.1, 0.7]:
            dropped_var = dropout(var, kprob, tf.constant(True))
            dropped_size = tf.reduce_sum(tf.to_int32(tf.equal(dropped_var, 0.0)))

            dsize = s.run(dropped_size)

            expected_dropped_size = 10000 * (1 - kprob)

            self.assertTrue(np.isclose(expected_dropped_size, dsize, atol=100))
            
    def test_train_false(self):
        """Checks that dropout does not """

        var = tf.ones([10000])
        s = tf.Session()

        dropped_var = dropout(var, 0.1, tf.constant(False))
        dropped_size = tf.reduce_sum(dropped_var)
        dsize = s.run(dropped_size)

        self.assertTrue(np.equal(10000, dsize, atol=100))


if __name__ == "__main__":
    unittest.main()
