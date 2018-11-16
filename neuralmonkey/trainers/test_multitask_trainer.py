#!/usr/bin/env python3.5
"""Unit tests for the multitask trainer."""
# pylint: disable=comparison-with-callable,attribute-defined-outside-init

import unittest
import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.trainers.generic_trainer import Objective, GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.decorators import tensor


class TestMP(ModelPart):

    @tensor
    def loss(self):
        self.var = tf.get_variable(name="var", shape=[], dtype=tf.float32)
        return 10 - self.var


class TestMultitaskTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

    def setUp(self):
        self.mpart = TestMP("dummy_model_part")
        self.mpart_2 = TestMP("dummy_model_part_2")

        objective = Objective(
            name="dummy_objective",
            decoder=self.mpart,
            loss=self.mpart.loss,
            gradients=None,
            weight=None)

        objective_2 = Objective(
            name="dummy_objective_2",
            decoder=self.mpart_2,
            loss=self.mpart_2.loss,
            gradients=None,
            weight=None)

        self.trainer1 = GenericTrainer([objective])
        self.trainer2 = GenericTrainer([objective_2], clip_norm=1.0)

    def test_mt_trainer(self):

        trainer = MultitaskTrainer(
            [self.trainer1, self.trainer2, self.trainer1])

        self.assertSetEqual(trainer.feedables, {self.mpart, self.mpart_2})
        self.assertSetEqual(trainer.parameterizeds, {self.mpart, self.mpart_2})

        self.assertSetEqual(
            set(trainer.var_list), {self.mpart.var, self.mpart_2.var})

        self.assertTrue(trainer.trainer_idx == 0)

        executable = trainer.get_executable()
        mparts, fetches, feeds = executable.next_to_execute()
        self.assertSetEqual(mparts, {self.mpart})
        self.assertFalse(feeds[0])

        self.assertTrue(trainer.trainer_idx == 1)
        self.assertTrue(fetches["losses"][0] == self.mpart.loss)

        executable = trainer.get_executable()
        mparts, fetches, feeds = executable.next_to_execute()
        self.assertSetEqual(mparts, {self.mpart_2})
        self.assertFalse(feeds[0])

        self.assertTrue(trainer.trainer_idx == 2)
        self.assertTrue(fetches["losses"][0] == self.mpart_2.loss)

        executable = trainer.get_executable()
        mparts, fetches, feeds = executable.next_to_execute()
        self.assertSetEqual(mparts, {self.mpart})
        self.assertFalse(feeds[0])

        self.assertTrue(trainer.trainer_idx == 0)
        self.assertTrue(fetches["losses"][0] == self.mpart.loss)


if __name__ == "__main__":
    unittest.main()
