"""CNN for image processing."""

from typing import List, Tuple, Type, Optional

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import conv2d, max_pool2d

from neuralmonkey.checking import assert_shape
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.decoding_function import Attention
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.nn.projection import multilayer_projection


class CNNEncoder(ModelPart, Attentive):
    """An image encoder.

    It projects the input image through a serie of convolutioal operations. The
    projected image is vertically cut and fed to stacked RNN layers which
    encode the image into a single vector.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self,
                 name: str,
                 data_id: str,
                 convolutions: List[Tuple[int, int, Optional[int]]],
                 image_height: int, image_width: int, pixel_dim: int,
                 fully_connected: Optional[List[int]] = None,
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
            dropout_keep_prob: Probability of keeping neurons active in
                dropout. Dropout is done between all convolutional layers and
                fully connected layer.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        self._image_height = image_height
        self._image_width = image_width
        self._pixel_dim = pixel_dim
        self._convolutions = convolutions
        self._fully_connected = fully_connected

        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")
        self.image_processing_layers = []  # type: List[tf.Tensor]

    @tensor
    def image_input(self):
        return tf.placeholder(
            tf.float32,
            shape=(None, self._image_height, self._image_width,
                   self._pixel_dim),
            name="input_images")

    @tensor
    def states(self):
        last_layer = self.image_input

        with tf.variable_scope("convolutions"):
            for i, (filter_size,
                    n_filters,
                    pool_size) in enumerate(self._convolutions):
                with tf.variable_scope("cnn_layer_{}".format(i)):
                    last_layer = conv2d(last_layer, n_filters, filter_size)
                    self.image_processing_layers.append(last_layer)

                    if pool_size:
                        last_layer = max_pool2d(last_layer, pool_size)
                        self.image_processing_layers.append(last_layer)

        return last_layer

    @tensor
    def encoded(self):
        # pylint: disable=no-member
        last_height, last_width, last_n_channels = [
            s.value for s in self.states.get_shape()[1:]]
        # pylint: enable=no-member

        if self._fully_connected is None:
            # we average out by the image size -> shape is number
            # channels from the last convolution
            encoded = tf.reduce_mean(self.states, [1, 2])
            assert_shape(encoded, [None, self._convolutions[-1][1]])
            return encoded

        states_flat = tf.reshape(
            self.states,
            [-1, last_width * last_height * last_n_channels])
        return multilayer_projection(
            states_flat, self._fully_connected,
            activation=tf.nn.relu,
            dropout_keep_prob=self.dropout_keep_prob,
            train_mode=self.train_mode)

    @tensor
    def _attention_tensor(self) -> tf.Tensor:
        # pylint: disable=no-member
        last_height, last_width, last_n_channels = [
            s.value for s in self.states.get_shape()[1:]]
        # pylint: enable=no-member
        return tf.reshape(
            self.states,
            [-1, last_width * last_height, last_n_channels])

    @tensor
    def _attention_mask(self) -> tf.Tensor:
        return tf.ones(tf.shape(self._attention_tensor)[:2])

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        # if it is from the pickled file, it is list, not numpy tensor,
        # so convert it as as a prevention
        images = np.array(dataset.get_series(self.data_id))

        f_dict = {}
        f_dict[self.image_input] = images / 225.0

        f_dict[self.train_mode] = train
        return f_dict
