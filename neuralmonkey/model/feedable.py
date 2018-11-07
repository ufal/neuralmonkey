from abc import ABCMeta
from typing import Any, Dict

import tensorflow as tf

from neuralmonkey.dataset import Dataset

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Any]
# pylint: enable=invalid-name


# pylint: disable=too-few-public-methods
# TODO add some public methods
class Feedable(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
        self.batch_size = tf.placeholder(tf.int32, [], "batch_size")

    def feed_dict(self, dataset: Dataset, train: bool = True) -> FeedDict:
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        fd[self.batch_size] = len(dataset)
        return fd
