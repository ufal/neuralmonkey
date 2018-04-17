from typing import List, Optional
from typeguard import check_argument_types

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import Stateful, SpatialStatefulWithOutput
from neuralmonkey.tf_utils import get_variable


# pylint: disable=too-few-public-methods


class StatefulFiller(ModelPart, Stateful):
    """Placeholder class for stateful input.

    This model part is used to feed 1D tensors to the model. Optionally, it
    projects the states to given dimension.
    """

    def __init__(self,
                 name: str,
                 dimension: int,
                 data_id: str,
                 output_shape: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Instantiate StatefulFiller.

        Args:
            name: Name of the model part.
            dimension: Dimensionality of the input.
            data_id: Series containing the numpy objects.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)
        check_argument_types()

        if dimension <= 0:
            raise ValueError("Input vector dimension must be positive.")
        if output_shape is not None and output_shape <= 0:
            raise ValueError("Output vector dimension must be positive.")

        self.vector = tf.placeholder(
            tf.float32, shape=[None, dimension])
        self.data_id = data_id

        with self.use_scope():
            if output_shape is not None and dimension != output_shape:
                project_w = get_variable(
                    shape=[dimension, output_shape],
                    name="img_init_proj_W")
                project_b = get_variable(
                    name="img_init_b", shape=[output_shape],
                    initializer=tf.zeros_initializer())

                self._encoded = tf.matmul(
                    self.vector, project_w) + project_b
            else:
                self._encoded = self.vector

    @property
    def output(self) -> tf.Tensor:
        return self._encoded

    # pylint: disable=unused-argument
    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.vector: dataset.get_series(self.data_id)}
    # pylint: enable=unused-argument


class SpatialFiller(ModelPart, SpatialStatefulWithOutput):
    """Placeholder class for 3D numerical input.

    This model part is used to feed 3D tensors (e.g., pre-trained convolutional
    maps image captioning). Optionally, the states are projected to given size.
    """

    def __init__(self,
                 name: str,
                 input_shape: List[int],
                 data_id: str,
                 projection_dim: int = None,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None,
                 initializers: InitializerSpecs = None) -> None:
        """Instantiate SpatialFiller.

        Args:
            name: Name of the model part.
            input_shape: Dimensionality of the input.
            data_id: Name of the data series with numpy objects.
            projection_dim: Optional, dimension of the states projection.
        """
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        assert len(input_shape) == 3

        self.data_id = data_id
        self.input_shape = input_shape
        self.projection_dim = projection_dim

        features_shape = [None] + self.input_shape  # type: ignore
        with self.use_scope():
            self.spatial_input = tf.placeholder(
                tf.float32, shape=features_shape, name="spatial_states")

    @tensor
    def output(self) -> tf.Tensor:
        return tf.reduce_mean(
            self.spatial_states, axis=[1, 2], name="average_image")

    @tensor
    def spatial_states(self) -> tf.Tensor:
        if self.projection_dim:
            return tf.layers.conv2d(
                self.spatial_input, filters=self.projection_dim,
                kernel_size=1, activation=None)
        return self.spatial_input

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        return tf.ones(tf.shape(self.spatial_states)[:3])

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.spatial_input: dataset.get_series(self.data_id)}
