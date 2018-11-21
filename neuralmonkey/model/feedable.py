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
    """Base class for feedable model parts.

    In TensorFlow, data is provided to the model using placeholders. Neural
    Monkey abstraction objects, such as encoders or decoders, can be members of
    this class in order to be able to receive data inputs from the framework.

    All feedable objects have a `feed_dict` method, which gets the current
    dataset and returns a `FeedDict` dictionary which assigns values to
    symbolic placeholders.

    Additionally, each Feedable object has two placeholders which are fed
    automatically in this super class - `batch_size` and `train_mode`.
    """

    def __init__(self) -> None:
        self.train_mode = tf.placeholder(tf.bool, [], "train_mode")
        self.batch_size = tf.placeholder(tf.int32, [], "batch_size")

    def feed_dict(self, dataset: Dataset, train: bool = True) -> FeedDict:
        """Return a feed dictionary for the given feedable object.

        Arguments:
            dataset: A dataset instance from which to get the data.
            train: Boolean indicating whether the model runs in training mode.

        Returns:
            A `FeedDict` dictionary object.
        """
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        fd[self.batch_size] = len(dataset)
        return fd
