from typing import List, Optional

import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict

# tests: lint, mypy

# pylint: disable=too-few-public-methods


class VectorEncoder(ModelPart):

    def __init__(self,
                 name: str,
                 dimension: int,
                 output_shape: int,
                 data_id: str,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        self.image_features = tf.placeholder(
            tf.float32, shape=[None, dimension])
        self.dimension = dimension
        self.output_shape = output_shape
        self.data_id = data_id

        self.flat = self.image_features

        project_w = tf.get_variable(
            shape=[dimension, output_shape],
            name="img_init_proj_W")
        project_b = tf.get_variable(
            name="img_init_b",
            initializer=tf.zeros_initializer([output_shape]))

        self.encoded = tf.tanh(tf.matmul(self.flat, project_w) + project_b)

    # pylint: disable=unused-argument
    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        return {self.image_features: dataset.get_series(self.data_id)}


class PostCNNImageEncoder(ModelPart, Attentive):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 input_shape: List[int],
                 output_shape: int,
                 data_id: str,
                 dropout_keep_prob: float=1.0,
                 attention_type=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        assert len(input_shape) == 3
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(attention_type, {})

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob
        self.attention_type = attention_type

        with tf.variable_scope(self.name):
            self.dropout_placeholder = tf.placeholder(tf.float32)
            features_shape = [None] + input_shape  # type: ignore
            self.image_features = tf.placeholder(tf.float32,
                                                 shape=features_shape,
                                                 name="image_input")

            self.flat = tf.reduce_mean(self.image_features,
                                       reduction_indices=[1, 2],
                                       name="average_image")
            project_w = tf.get_variable(
                name="img_init_proj_W",
                shape=[input_shape[2], output_shape],
                initializer=tf.random_normal_initializer())
            project_b = tf.get_variable(
                name="img_init_b",
                initializer=tf.zeros_initializer([output_shape]))

            self.encoded = tf.tanh(tf.matmul(self.flat, project_w) + project_b)

            self.__attention_tensor = tf.reshape(
                self.image_features,
                [-1, input_shape[0] * input_shape[1],
                 input_shape[2]],
                name="flatten_image")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        res = {}  # type: FeedDict
        res[self.image_features] = dataset.get_series(self.data_id)

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            res[self.dropout_placeholder] = 1.0

        return res
