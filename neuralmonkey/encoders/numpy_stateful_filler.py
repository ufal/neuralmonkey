# TODO untested module
from typing import Dict, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import Stateful, SpatialStatefulWithOutput


# pylint: disable=too-few-public-methods
class StatefulFiller(ModelPart, Stateful):
    """Placeholder class for stateful input.

    This model part is used to feed 1D tensors to the model. Optionally, it
    projects the states to given dimension.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 dimension: int,
                 data_id: str,
                 output_shape: int = None,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Instantiate StatefulFiller.

        Arguments:
            name: Name of the model part.
            dimension: Dimensionality of the input.
            data_id: Series containing the numpy objects.
            output_shape: Dimension of optional state projection.
        """
        check_argument_types()
        ModelPart.__init__(
            self, name, reuse, save_checkpoint, load_checkpoint, initializers)

        self.data_id = data_id
        self.dimension = dimension
        self.output_shape = output_shape

        if self.dimension <= 0:
            raise ValueError("Input vector dimension must be positive.")
        if self.output_shape is not None and self.output_shape <= 0:
            raise ValueError("Output vector dimension must be positive.")
    # pylint: enable=too-many-arguments

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.float32}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        return {self.data_id: tf.TensorShape([None, self.dimension])}

    @tensor
    def vector(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def output(self) -> tf.Tensor:
        if self.output_shape is None or self.dimension == self.output_shape:
            return self.vector

        return tf.layers.dense(self.vector, self.output_shape)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)
        fd[self.vector] = dataset.get_series(self.data_id)
        return fd


class SpatialFiller(ModelPart, SpatialStatefulWithOutput):
    """Placeholder class for 3D numerical input.

    This model part is used to feed 3D tensors (e.g., pre-trained convolutional
    maps image captioning). Optionally, the states are projected to given size.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_shape: List[int],
                 data_id: str,
                 projection_dim: int = None,
                 ff_hidden_dim: int = None,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Instantiate SpatialFiller.

        Args:
            name: Name of the model part.
            input_shape: Dimensionality of the input.
            data_id: Name of the data series with numpy objects.
            projection_dim: Optional, dimension of the states projection.
        """
        check_argument_types()
        ModelPart.__init__(
            self, name, reuse, save_checkpoint, load_checkpoint, initializers)

        self.data_id = data_id
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.ff_hidden_dim = ff_hidden_dim

        if self.ff_hidden_dim is not None and self.projection_dim is None:
            raise ValueError(
                "projection_dim must be provided when using ff_hidden_dim")

        if len(self.input_shape) != 3:
            raise ValueError("The input shape should have 3 dimensions.")
    # pylint: enable=too-many-arguments

    @property
    def input_types(self) -> Dict[str, tf.DType]:
        return {self.data_id: tf.float32}

    @property
    def input_shapes(self) -> Dict[str, tf.TensorShape]:
        shape = [None] + self.input_shape  # type: ignore
        return {self.data_id: tf.TensorShape(shape)}

    @tensor
    def spatial_input(self) -> tf.Tensor:
        return self.dataset[self.data_id]

    @tensor
    def output(self) -> tf.Tensor:
        return tf.reduce_mean(
            self.spatial_states, axis=[1, 2], name="average_image")

    @tensor
    def spatial_states(self) -> tf.Tensor:
        if self.ff_hidden_dim:
            projected = tf.layers.conv2d(
                self.spatial_input, filters=self.ff_hidden_dim,
                kernel_size=1, activation=tf.nn.relu)
        else:
            projected = self.spatial_input

        if self.projection_dim:
            return tf.layers.conv2d(
                projected, filters=self.projection_dim,
                kernel_size=1, activation=None)

        # pylint: disable=comparison-with-callable
        assert projected == self.spatial_input
        # pylint: enable=comparison-with-callable

        return self.spatial_input

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        return tf.ones(tf.shape(self.spatial_states)[:3])

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)
        fd[self.spatial_input] = list(dataset.get_series(self.data_id))
        return fd
