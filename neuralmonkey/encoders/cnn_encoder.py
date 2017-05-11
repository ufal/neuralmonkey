"""CNN for image processing."""

from typing import List, Tuple, Type, Optional

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import conv2d, max_pool2d, batch_norm

from neuralmonkey.checking import assert_shape
from neuralmonkey.dataset import Dataset
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoding_function import Attention
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.projection import multilayer_projection
from neuralmonkey.nn.utils import dropout


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
        train_mode: Placeholder for boolean telleing whether the training
            is running.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self,
                 name: str,
                 data_id: str,
                 convolutions: List[Tuple[int, int, Optional[int]]],
                 image_height: int, image_width: int, pixel_dim: int,
                 fully_connected: Optional[List[int]] = None,
                 batch_normalization: bool = True,
                 local_response_normalization: bool = True,
                 dropout_keep_prob: float = 0.5,
                 attention_type: Type = Attention,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:
        """Initialize a convolutional network for image processing.

        Args:
            convolutions: Configuration of convolutional layers. It is a list
                of triplets of integers where the values are: size of the
                convolutional window, number of convolutional filters, and size
                of max-pooling window. If the max-pooling size is set to None,
                no pooling is performed.
            data_id: Identifier of the data series in the dataset.
            image_height: Height of the input image in pixels.
            image_width: Width of the image.
            pixel_dim: Number of color channels in the input images.
            batch_normalization: Flag whether the batch normalization
                should be used between the convolutional layers.
            local_response_normalization: Flag whether to use local
                response normalization between the convolutional layers.
            dropout_keep_prob: Probability of keeping neurons active in
                dropout. Dropout is done between all convolutional layers and
                fully connected layer.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        with self.use_scope():
            self.train_mode = tf.placeholder(tf.bool, shape=[],
                                             name="train_mode")
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

            self.image_processing_layers = []  # type: List[tf.Tensor]

            with tf.variable_scope("convolutions"):
                for i, (filter_size,
                        n_filters,
                        pool_size) in enumerate(convolutions):
                    with tf.variable_scope("cnn_layer_{}".format(i)):
                        last_layer = conv2d(last_layer, n_filters, filter_size)
                        self.image_processing_layers.append(last_layer)

                        if pool_size:
                            last_layer = max_pool2d(last_layer, pool_size)
                            self.image_processing_layers.append(last_layer)
                            last_padding_masks = max_pool2d(
                                last_padding_masks, pool_size)

                        if local_response_normalization:
                            last_layer = tf.nn.local_response_normalization(
                                last_layer)

                        if batch_normalization:
                            last_layer = batch_norm(
                                last_layer, is_training=self.train_mode)

                        last_layer = dropout(last_layer, dropout_keep_prob,
                                             self.train_mode)

                # last_layer shape is batch X height X width X channels
                last_layer = last_layer * last_padding_masks

            # pylint: disable=no-member
            last_height, last_width, last_n_channels = [
                s.value for s in last_layer.get_shape()[1:]]
            # pylint: enable=no-member

            if fully_connected is None:
                # we average out by the image size -> shape is number
                # channels from the last convolution
                self.encoded = tf.reduce_mean(last_layer, [1, 2])
                assert_shape(self.encoded, [None, convolutions[-1][1]])
            else:
                last_layer_flat = tf.reshape(
                    last_layer,
                    [-1, last_width * last_height * last_n_channels])
                self.encoded = multilayer_projection(
                    last_layer_flat, fully_connected,
                    activation=tf.nn.relu,
                    dropout_keep_prob=dropout_keep_prob,
                    train_mode=self.train_mode)

            self.__attention_tensor = tf.reshape(
                last_layer, [-1, last_width * last_height, last_n_channels])

            self.__attention_mask = tf.reshape(
                last_padding_masks, [-1, last_width * last_height])

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self.__attention_tensor

    @property
    def _attention_mask(self) -> tf.Tensor:
        return self.__attention_mask

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        # if it is from the pickled file, it is list, not numpy tensor,
        # so convert it as as a prevention
        images = np.array(dataset.get_series(self.data_id))

        f_dict = {}
        f_dict[self.input_op] = images / 225.0

        # it is one everywhere where non-zero, i.e. zero columns are masked out
        f_dict[self.padding_masks] = \
            np.sum(np.sign(images), axis=3, keepdims=True)

        f_dict[self.train_mode] = train
        return f_dict
