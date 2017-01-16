"""CNN for image processing."""

from typing import List, Tuple, Type, Optional

import numpy as np
import tensorflow as tf

from neuralmonkey.checking import assert_shape
from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoding_function import Attention
from neuralmonkey.model.model_part import ModelPart, FeedDict

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes, too-few-public-methods


class CNNEncoder(ModelPart, Attentive):
    """An image encoder.

    It projects the input image through a serie of convolutioal operations. The
    projected image is vertically cut and fed to stacked RNN layers which
    encode the image into a single vector.

    Attributes:

        input_op: Placeholder for the batch of input images
        padding_masks: Placeholder for matrices capturing telling where the
            image has been padded.
        image_processing_layers: List of TensorFlow operator that are
            visualizable image transformations.
        encoded: Operator that returns a batch of ecodede image (intended
            as an input for the decoder).
        attention_tensor: Tensor computing a batch of attention
            matrices for the decoder.
        is_training: Placeholder for boolean telleing whether the training
            is running.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self,
                 name: str,
                 data_id: str,
                 convolutions: List[Tuple[int, int, Optional[int]]],
                 image_height: int, image_width: int, pixel_dim: int,
                 batch_normalization: bool=True,
                 local_response_normalization: bool=True,
                 dropout_keep_prob: float=0.5,
                 attention_type: Type=Attention,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        """Initialize a convolutional network for image processing.

        Args:
            convolutions (list): Configuration convolutional layers. It is a
                list of tripplets of integers where the values are: size of the
                convolutional window, number of convolutional filters, and size
                of max-pooling window. If the max-pooling size is set to None,
                no pooling is performed.
            data_id: Identifier of the data series in the dataset.
            image_height: Height of the input image in pixels.
            image_width: Width of the images (padded)
            pixel_dim: Number of color channels in the input images.
            batch_normalization: Flag whether the batch normalization
                should be used between the convolutional layers.
            local_response_normalization: Flag whether to use local
                response normalization between the convolutional layers.
            dropout_placeholder: Placeholder keeping the
                dropout keeping probability

        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.convolutions = convolutions
        self.data_id = data_id
        self.image_height = image_height
        self.image_width = image_width
        self.pixel_dim = pixel_dim
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope(name):
            self.dropout_placeholder = tf.placeholder(
                tf.float32, name="dropout")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.input_op = tf.placeholder(
                tf.float32,
                shape=(None, image_height, image_width, pixel_dim),
                name="input_images")

            self.padding_masks = tf.placeholder(
                tf.float32,
                shape=(None, image_height, image_width, 1),
                name="padding_masks")

            last_layer = self.input_op
            last_padding_masks = self.padding_masks
            last_n_channels = pixel_dim

            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.image_processing_layers = []  # type: List[tf.Tensor]

            with tf.variable_scope("convolutions"):
                for i, (filter_size,
                        n_filters,
                        pool_size) in enumerate(convolutions):
                    with tf.variable_scope("cnn_layer_{}".format(i)):
                        last_layer = _convolution(
                            last_layer, last_n_channels, filter_size,
                            n_filters)
                        last_n_channels = n_filters
                        self.image_processing_layers.append(last_layer)

                        if pool_size:
                            # TODO do the pooling properly
                            last_layer = tf.nn.max_pool(
                                last_layer, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
                            last_padding_masks = tf.nn.max_pool(
                                last_padding_masks, [1, 2, 2, 1],
                                [1, 2, 2, 1], "SAME")
                            self.image_processing_layers.append(last_layer)
                            assert image_height % 2 == 0
                            image_height //= 2
                            assert image_width % 2 == 0
                            image_width //= 2

                        if local_response_normalization:
                            last_layer = tf.nn.local_response_normalization(
                                last_layer)

                        if batch_normalization:
                            last_layer = _batch_norm(
                                last_layer, n_filters, self.is_training)

                        last_layer = tf.nn.dropout(
                            last_layer, keep_prob=self.dropout_placeholder)

                # last_layer shape is batch X height X width X channels
                last_layer = last_layer * last_padding_masks

            # we average out by the image size -> shape is number
            # channels from the last convolution
            self.encoded = tf.reduce_mean(last_layer, [1, 2])
            # TODO assert shape
            assert_shape(self.encoded, [None, self.convolutions[-1][1]])

            self.__attention_tensor = tf.reshape(
                last_layer, [-1, image_width,
                             last_n_channels * image_height])

            self.__attention_mask = tf.squeeze(
                tf.reduce_prod(last_padding_masks, [1]), [2])

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self.__attention_tensor

    @property
    def _attention_mask(self) -> tf.Tensor:
        return self.__attention_mask

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        # if it is from the pickled file, it is list, not numpy tensor,
        # so convert it as as a prevention
        images = np.array(dataset.get_series(self.data_id))

        f_dict = {}
        f_dict[self.input_op] = images / 225.0

        # it is one everywhere where non-zero, i.e. zero columns are masked out
        f_dict[self.padding_masks] = \
            np.sum(np.sign(images), axis=3, keepdims=True)

        if train:
            f_dict[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            f_dict[self.dropout_placeholder] = 1.0
        f_dict[self.is_training] = train
        return f_dict


def _convolution(last_layer: tf.Tensor, last_n_channels: int,
                 filter_size: int, n_filters: int) -> tf.Tensor:
    """Applies convolution on a filter bank."""
    conv_w = tf.get_variable(
        "wieghts",
        shape=[filter_size, filter_size, last_n_channels, n_filters],
        initializer=tf.truncated_normal_initializer(stddev=.1))
    conv_b = tf.get_variable("biases", shape=[n_filters],
                             initializer=tf.constant_initializer(.1))
    conv_activation = tf.nn.conv2d(
        last_layer, conv_w, [1, 1, 1, 1], "SAME") + conv_b
    assert_shape(conv_activation, [
        None,
        last_layer.get_shape()[1].value,
        last_layer.get_shape()[2].value,
        filter_size])
    return tf.nn.relu(conv_activation)


# pylint: disable=too-many-locals


def _batch_norm(tensor, n_out, phase_train, scope='bn', scale_after_norm=True):
    """ Batch normalization on convolutional maps.

    Taken from
    http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Arguments:
        tensor:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        rescale_after_norm: whether to rescale the normalization output

    Return:
        normed:      batch-normalized maps

    """

    with tf.variable_scope(scope):
        beta = tf.get_variable(
            name='beta', initializer=tf.zeros_initializer(shape=[n_out]))
        gamma = tf.get_variable(
            name="gamma", initializer=tf.ones_initializer(shape=[n_out]),
            trainable=scale_after_norm)

        batch_mean, batch_var = tf.nn.moments(
            tensor, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(
            tensor, mean, var, beta, gamma, 1e-3, scale_after_norm)
        return normed
