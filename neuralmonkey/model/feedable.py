from abc import ABCMeta

from typing import Any, Dict, List
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf
from neuralmonkey.dataset import Dataset

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Any]
# pylint: enable=invalid-name


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
        self._dataset = None  # type: Optional[Dict[str, tf.Tensor]]

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

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {}

    @property
    def input_shapes(self) -> Dict[str, List[int]]:
        return {}

    @property
    def dataset(self) -> Dict[str, tf.Tensor]:
        if self._dataset is None:
            raise RuntimeError("Getting dataset before registering it.")
        return self._dataset

    def register_input(self, dataset: Dict[str, tf.Tensor]) -> None:
        self._dataset = dataset
