#!/usr/bin/env python3.5
"""Unit tests for functions.py."""

import unittest
import tensorflow as tf

from neuralmonkey.functions import piecewise_function


class TestPiecewiseFunction(unittest.TestCase):

    def test_piecewise_constant(self):
        x = tf.placeholder(dtype=tf.int32)
        y = piecewise_function(x, [-0.5, 1.2, 3, 2], [-1, 2, 1000],
                               dtype=tf.float32)

        with tf.Session() as sess:
            self.assertAlmostEqual(sess.run(y, {x: -2}), -0.5)
            self.assertAlmostEqual(sess.run(y, {x: -1}), 1.2)
            self.assertAlmostEqual(sess.run(y, {x: 999}), 3)
            self.assertAlmostEqual(sess.run(y, {x: 1000}), 2)
            self.assertAlmostEqual(sess.run(y, {x: 1001}), 2)


if __name__ == "__main__":
    unittest.main()
