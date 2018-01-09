#!/usr/bin/env python3.5

import os
import shutil
import tempfile
import unittest
import numpy as np
import tensorflow as tf

from neuralmonkey.tf_manager import TensorFlowManager


class TestTensorFlowManager(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        tf.get_variable("x", shape=[])

    def tearDown(self):
        tf.reset_default_graph()
        shutil.rmtree(self.tmp_dir)

    def test_early_stopping(self):
        tf_manager = TensorFlowManager(num_sessions=1, num_threads=2,
                                       patience=5, save_n_best=3)
        tf_manager.init_saving(os.path.join(self.tmp_dir, "vars"))

        scores = [1., 3., 10., 8., 7., 8., 6., 7., 9.]
        for i, score in enumerate(scores[:-1]):
            tf_manager.validation_hook(score, epoch=1, batch=i)
            self.assertFalse(tf_manager.should_stop)

        tf_manager.validation_hook(scores[-1], epoch=1, batch=len(scores) - 1)
        self.assertTrue(tf_manager.should_stop)
        self.assertEqual(tf_manager.best_score_epoch, 1)
        self.assertEqual(tf_manager.best_score_batch, np.argmax(scores))

    def test_no_early_stopping(self):
        tf_manager = TensorFlowManager(num_sessions=1, num_threads=2)
        tf_manager.init_saving(os.path.join(self.tmp_dir, "vars"))

        for i, score in enumerate(range(500, 0, -1)):
            tf_manager.validation_hook(score, epoch=1, batch=i)
            self.assertFalse(tf_manager.should_stop)


if __name__ == "__main__":
    unittest.main()
