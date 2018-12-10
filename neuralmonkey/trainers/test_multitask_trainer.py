#!/usr/bin/env python3.5
"""Unit tests for the multitask trainer."""
# pylint: disable=comparison-with-callable,attribute-defined-outside-init

import unittest
import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.logging import log
from neuralmonkey.trainers.generic_trainer import Objective, GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.decorators import tensor


class TestMP(ModelPart):

    # pylint: disable=no-self-use
    @tensor
    def var(self) -> tf.Variable:
        return tf.get_variable(name="var", shape=[], dtype=tf.float32)
    # pylint: enable=no-self-use

    @tensor
    def loss(self) -> tf.Tensor:
        return 10 - self.var


# pylint: disable=too-few-public-methods
class DummyObjective(Objective[TestMP]):
    def __init__(self, name: str, decoder: TestMP) -> None:
        Objective[TestMP].__init__(self, name, decoder)

    @tensor
    def loss(self) -> tf.Tensor:
        return self.decoder.loss
# pylint: enable=too-few-public-methods


class TestMultitaskTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()

    def setUp(self):
        self.mpart = TestMP("dummy_model_part")
        self.mpart_2 = TestMP("dummy_model_part_2")

        objective = DummyObjective(name="dummy", decoder=self.mpart)
        objective_2 = DummyObjective(name="dummy_2", decoder=self.mpart_2)

        self.trainer1 = GenericTrainer([objective])
        self.trainer2 = GenericTrainer([objective_2], clip_norm=1.0)

    def test_mt_trainer(self):
        # TODO multitask trainer is likely broken by changes in tf-data branch

        trainer = MultitaskTrainer(
            [self.trainer1, self.trainer2, self.trainer1])

        log("Blessing trainer fetches: {}".format(trainer.fetches))

        self.assertSetEqual(trainer.feedables, {self.mpart, self.mpart_2,
                                                self.trainer1, self.trainer2})
        self.assertSetEqual(trainer.parameterizeds, {self.mpart, self.mpart_2})

        self.assertSetEqual(
            set(trainer.var_list), {self.mpart.var, self.mpart_2.var})

        self.assertTrue(trainer.trainer_idx == 0)

        executable = trainer.get_executable()
        # mparts = trainer.feedables
        fetches, feeds = executable.next_to_execute()
        # self.assertSetEqual(mparts, {self.mpart})
        self.assertFalse(feeds)

        self.assertTrue(trainer.trainer_idx == 1)
        self.assertTrue(fetches["losses"][0] == self.mpart.loss)

        executable = trainer.get_executable()
        fetches, feeds = executable.next_to_execute()
        # self.assertSetEqual(mparts, {self.mpart_2})
        self.assertFalse(feeds)

        self.assertTrue(trainer.trainer_idx == 2)
        self.assertTrue(fetches["losses"][0] == self.mpart_2.loss)

        executable = trainer.get_executable()
        fetches, feeds = executable.next_to_execute()
        # self.assertSetEqual(mparts, {self.mpart})
        self.assertFalse(feeds)

        self.assertTrue(trainer.trainer_idx == 0)
        self.assertTrue(fetches["losses"][0] == self.mpart.loss)


if __name__ == "__main__":
    unittest.main()
