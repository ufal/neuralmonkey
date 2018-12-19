from abc import ABCMeta

from typing import Any, Dict, List
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf
from neuralmonkey.decorators import tensor

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
        self._dataset = None  # type: Optional[tf.data.Dataset]

    @tensor
    def batch_size(self) -> tf.Tensor:
        first_tensor = tf.contrib.framework.nest.flatten(self.dataset)[0]
        return tf.shape(first_tensor)[0]

    def feed_dict(self, train: bool = True) -> FeedDict:
        """Return a feed dictionary for the given feedable object.

        Arguments:
            train: Boolean indicating whether the model runs in training mode.

        Returns:
            A `FeedDict` dictionary object.
        """
        return {self.train_mode: train}

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
