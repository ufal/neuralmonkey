from typing import Callable, List, Optional
from typeguard import check_argument_types

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict


# pylint: disable=too-few-public-methods


class VectorEncoder(ModelPart):

    def __init__(self,
                 name: str,
                 dimension: int,
                 data_id: str,
                 output_shape: int = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        assert check_argument_types()

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

    # pylint: disable=unused-argument
    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.vector: dataset.get_series(self.data_id)}


class PostCNNImageEncoder(ModelPart, Attentive):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_shape: List[int],
                 output_shape: int,
                 data_id: str,
                 attention_type: Callable = None,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)
        assert check_argument_types()

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
            project_w = tf.get_variable(
                name="img_init_proj_W",
                shape=[input_shape[2], output_shape],
                initializer=tf.random_normal_initializer())
            project_b = tf.get_variable(
                name="img_init_b", shape=[output_shape],
                initializer=tf.zeros_initializer())

            self.encoded = tf.tanh(tf.matmul(self.flat, project_w) + project_b)

            self.__attention_tensor = tf.reshape(
                self.image_features,
                [-1, input_shape[0] * input_shape[1],
                 input_shape[2]],
                name="flatten_image")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        res = {}  # type: FeedDict
        res[self.image_features] = dataset.get_series(self.data_id)

        return res
