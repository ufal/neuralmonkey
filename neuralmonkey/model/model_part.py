"""Basic functionality of all model parts."""

from abc import ABCMeta, abstractproperty
from typing import Optional

import tensorflow as tf

from neuralmonkey.logging import log


class ModelPart(metaclass=ABCMeta):
    """Base class of all model parts."""
    def __init__(self,
                 name: str,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None):
        self._name = name
        self._save_checkpoint = save_checkpoint
        self._load_checkpoint = load_checkpoint

        self.__saver = None

    @property
    def name(self):
        """Name of the model part and its variable scope."""
        return self._name

    @abstractproperty
    def feed_dict(self, dataset):
        """Prepare feed dicts for part's placeholders from a dataset."""
        raise NotImplementedError("Abstract base class.")

    def __init_saver(self):
        if not self.__saver:
            with tf.variable_scope(self._name, reuse=True):
                parts_variables = tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope=self._name)
                self.__saver = tf.train.Saver(var_list=parts_variables)

    def save(self, session):
        """Save model part to a checkpoint file."""
        if self._save_checkpoint:
            self.__init_saver()
            self.__saver.save(session, self._save_checkpoint)

            log("Variables of '{}' saved to '{}'".format(
                self.name, self._save_checkpoint))

    def load(self, session):
        """Load model part from a checkpoint file."""
        if self._load_checkpoint:
            self.__init_saver()
            self.__saver.restore(session, self._load_checkpoint)

            log("Variables of '{}' loaded from '{}'".format(
                self.name, self._save_checkpoint))
