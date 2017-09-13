from typing import List, Optional
from typeguard import check_argument_types

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.model.stateful import Stateful, SpatialStatefulWithOutput


# pylint: disable=too-few-public-methods


class VectorEncoder(ModelPart, Stateful):

    def __init__(self,
                 name: str,
                 dimension: int,
                 data_id: str,
                 output_shape: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        check_argument_types()

        if dimension <= 0:
            raise ValueError("Input vector dimension must be postive.")
        if output_shape is not None and output_shape <= 0:
            raise ValueError("Output vector dimension must be postive.")

        self.vector = tf.placeholder(
            tf.float32, shape=[None, dimension])
        self.data_id = data_id

        with self.use_scope():
            if output_shape is not None and dimension != output_shape:
                project_w = tf.get_variable(
                    shape=[dimension, output_shape],
                    name="img_init_proj_W")
                project_b = tf.get_variable(
                    name="img_init_b", shape=[output_shape],
                    initializer=tf.zeros_initializer())

                self.encoded = tf.matmul(
                    self.vector, project_w) + project_b
            else:
                self.encoded = self.vector

    @property
    def output(self) -> tf.Tensor:
        return self.encoded

    # pylint: disable=unused-argument
    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.vector: dataset.get_series(self.data_id)}


class PostCNNImageEncoder(ModelPart, SpatialStatefulWithOutput):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_shape: List[int],
                 output_shape: int,
                 data_id: str,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        assert len(input_shape) == 3
        if output_shape <= 0:
            raise ValueError("Output vector dimension must be postive.")

        self.data_id = data_id

        with self.use_scope():
            features_shape = [None] + input_shape  # type: ignore
            self.image_features = tf.placeholder(tf.float32,
                                                 shape=features_shape,
                                                 name="image_input")

            self.flat = tf.reduce_mean(self.image_features,
                                       axis=[1, 2],
                                       name="average_image")

            self.project_w = tf.get_variable(
                name="img_init_proj_W",
                shape=[input_shape[2], output_shape],
                initializer=tf.random_normal_initializer())
            self.project_b = tf.get_variable(
                name="img_init_b", shape=[output_shape],
                initializer=tf.zeros_initializer())

    @tensor
    def output(self) -> tf.Tensor:
        return tf.tanh(tf.matmul(self.flat, self.project_w) + self.project_b)

    @tensor
    def spatial_states(self) -> tf.Tensor:
        return self.image_features

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        return tf.ones(tf.shape(self.spatial_states)[:3])

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        res = {}  # type: FeedDict
        res[self.image_features] = dataset.get_series(self.data_id)

        return res
