"""Basic functionality of all model parts."""

from abc import ABCMeta
from typing import Any, Dict, Optional

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.logging import log

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Any]
# pylint: disable=invalid-name


class ModelPart(metaclass=ABCMeta):
    """Base class of all model parts."""
    def __init__(self,
                 name: str,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        self._name = name
        self._save_checkpoint = save_checkpoint
        self._load_checkpoint = load_checkpoint

        self._saver = None  # type: Optional[tf.train.Saver]

    @property
    def name(self) -> str:
        """Name of the model part and its variable scope."""
        return self._name

    def feed_dict(self, dataset: Dataset, train: bool) -> FeedDict:
        """Prepare feed dicts for part's placeholders from a dataset."""
        raise NotImplementedError("Abstract base class.")

    def _init_saver(self) -> None:
        if not self._saver:
            with tf.variable_scope(self._name, reuse=True):
                parts_variables = tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope=self._name)
                self._saver = tf.train.Saver(var_list=parts_variables)

    def save(self, session: tf.Session) -> None:
        """Save model part to a checkpoint file."""
        if self._save_checkpoint:
            self._init_saver()
            self._saver.save(session, self._save_checkpoint)

            log("Variables of '{}' saved to '{}'".format(
                self.name, self._save_checkpoint))

    def load(self, session: tf.Session) -> None:
        """Load model part from a checkpoint file."""
        if self._load_checkpoint:
            self._init_saver()
            self._saver.restore(session, self._load_checkpoint)

            log("Variables of '{}' loaded from '{}'".format(
                self.name, self._load_checkpoint))
