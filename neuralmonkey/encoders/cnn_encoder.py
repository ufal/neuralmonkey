"""CNN for image processing."""

from typing import List, Tuple, Optional, Set

import numpy as np
import tensorflow as tf

from neuralmonkey.checking import assert_shape
from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import SpatialStatefulWithOutput
from neuralmonkey.model.stateful import (SpatialStatefulWithOutput,
                                         TemporalStatefulWithOutput)
from neuralmonkey.nn.projection import multilayer_projection
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell


class CNNEncoder(ModelPart, SpatialStatefulWithOutput):
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
                 batch_normalize: bool = False,
                 dropout_keep_prob: float = 0.5,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None,
                 initializers: InitializerSpecs = None) -> None:
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
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        self.image_height = image_height
        self.image_width = image_width
        self.pixel_dim = pixel_dim
        self.convolutions = convolutions
        self.fully_connected = fully_connected
        self.batch_normalize = batch_normalize
    # pylint: enable=too-many-arguments, too-many-locals

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, shape=[], name="mode_placeholder")
    # pylint: enable=no-self-use

    @tensor
    def image_input(self) -> tf.Tensor:
        return tf.placeholder(
            tf.float32,
            shape=(None, self.image_height, self.image_width,
                   self.pixel_dim),
            name="input_images")

    @tensor
    def image_mask(self) -> tf.Tensor:
        return tf.placeholder(
            tf.float32,
            shape=(None, self.image_height, self.image_width, 1),
            name="input_mask")

    @tensor
    def image_processing_layers(self) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        """Do all convolutions and return the last conditional map.

        There is not dropout between the convolutional layers, by
        default the activation function is ReLU.
        """
        last_layer = self.image_input
        last_mask = self.image_mask
        last_channels = self.pixel_dim
        image_processing_layers = []  # type: List[Tuple[tf.Tensor, tf.Tensor]]

        with tf.variable_scope("convolutions"):
            for i, specification in enumerate(self.convolutions):
                if specification[0] == 'C':
                    if len(specification) != 5:
                        raise ValueError((
                            'Specification of a convolutional layer needs to '
                            'have 5 members: kernel size, stride, padding, '
                            '# output channels, was {}').format(specification))
                    kernel_size, stride, pad, out_channels = specification[1:]

                    if pad not in ['same', 'valid']:
                        raise ValueError(
                            ('Padding must be "same" or "valid", '
                             'was "{}" in layer {}.').format(pad, i + 1))

                    with tf.variable_scope('layer_{}_convolution'.format(i)):
                        last_layer = tf.layers.conv2d(
                            last_layer, out_channels, kernel_size,
                            activation=None, padding='same')
                        last_channels = out_channels

                        if self.batch_normalize:
                            # pylint: disable=cell-var-from-loop
                            last_layer = tf.layers.batch_normalization(
                                last_layer, training=self.train_mode)
                            # pylint: enable=cell-var-from-loop

                        last_layer = tf.nn.relu(last_layer)

                        last_mask = tf.layers.max_pooling2d(
                            last_mask, kernel_size, stride, padding=pad)

                    image_processing_layers.append((last_layer, last_mask))
                elif specification[0] == 'M':
                    if len(specification) != 4:
                        raise ValueError((
                            'Specification of a convolutional layer needs to '
                            'have 5 members: kernel size, stride, padding, '
                            'was {}').format(specification))
                    pool_size, stride, pad = specification[1:]

                    if pad not in ['same', 'valid']:
                        raise ValueError(
                            ('Padding must be "same" or "valid", '
                             'was "{}" in layer {}.').format(pad, i + 1))

                    with tf.variable_scope('layer_{}_max_pool'.format(i)):
                        last_layer = tf.layers.max_pooling2d(
                            last_layer, pool_size, 2)
                        last_mask = tf.layers.max_pooling2d(
                            last_mask, pool_size, 2)
                    image_processing_layers.append((last_layer, last_mask))
                elif specification[0] == 'R':
                    if not self.batch_normalize:
                        raise ValueError('Using ResNet blocks requires batch '
                                         'normalization to be turned on.')
                    kernel_size, out_channels = specification[1:]

                    with tf.variable_scope('layer_{}_resnet_block'.format(i)):
                        if out_channels == last_channels:
                            before_resnet_block = last_layer
                        else:
                            with tf.variable_scope('project_input'):
                                before_resnet_block = tf.layers.conv2d(
                                    last_layer, out_channels, 1, 1,
                                    'same', activation=None)
                                before_resnet_block = (
                                    tf.layers.batch_normalization(
                                        before_resnet_block,
                                        training=self.train_mode))
                            last_channels = out_channels

                        with tf.variable_scope('conv_a'):
                            last_layer = tf.layers.batch_normalization(
                                last_layer, training=self.train_mode)
                            last_layer = tf.nn.relu(last_layer)
                            last_layer = tf.layers.conv2d(
                                last_layer, last_channels, kernel_size,
                                padding='same',
                                activation=None)

                        with tf.variable_scope('conv_b'):
                            last_layer = tf.layers.batch_normalization(
                                last_layer, training=self.train_mode)
                            last_layer = tf.nn.relu(last_layer)
                            last_layer = tf.layers.conv2d(
                                last_layer, last_channels, kernel_size,
                                padding='same',
                                activation=None)

                        last_layer = last_layer + before_resnet_block

                    image_processing_layers.append((last_layer, last_mask))
                else:
                    raise ValueError(
                        'Unknown type of convoutional layer #{}: "{}"'.format(
                            i + 1, specification[0]))

        return image_processing_layers

    @tensor
    def spatial_states(self):
        # pylint: disable=unsubscriptable-object
        return self.image_processing_layers[-1][0]
        # pylint: enable=unsubscriptable-object

    @tensor
    def spatial_mask(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return self.image_processing_layers[-1][1]
        # pylint: enable=unsubscriptable-object

    @tensor
    def output(self) -> tf.Tensor:
        """Output vector of the CNN.

        If there are specified some fully connected layers, there are applied
        on top of the last convolutional map. Dropout is applied between all
        layers, default activation function is ReLU. There are only projection
        layers, no softmax is applied.

        If there is fully_connected layer specified, average-pooled last
        convolutional map is used as a vector output.
        """
        # pylint: disable=no-member
        last_height, last_width, last_n_channels = [
            s.value for s in self.spatial_states.get_shape()[1:]]
        # pylint: enable=no-member

        if self.fully_connected is None:
            # we average out by the image size -> shape is number
            # channels from the last convolution
            encoded = tf.reduce_mean(self.spatial_states, [1, 2])
            assert_shape(encoded, [None, self.convolutions[-1][1]])
            return encoded

        states_flat = tf.reshape(
            self.spatial_states,
            [-1, last_width * last_height * last_n_channels])
        return multilayer_projection(
            states_flat, self.fully_connected,
            activation=tf.nn.relu,
            dropout_keep_prob=self.dropout_keep_prob,
            train_mode=self.train_mode)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        # if it is from the pickled file, it is list, not numpy tensor,
        # so convert it as as a prevention
        images = np.array(dataset.get_series(self.data_id))

        f_dict = {}
        f_dict[self.image_input] = images / 225.0
        # it is one everywhere where non-zero, i.e. zero columns are masked out
        f_dict[self.image_mask] = np.sum(
            np.sign(images), axis=3, keepdims=True)

        f_dict[self.train_mode] = train
        return f_dict


class OCREncoder(ModelPart, TemporalStatefulWithOutput):
    """Apply bidirectionaly RNN over image processing CNN."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 cnn: CNNEncoder,
                 rnn_state_size: int,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None) -> None:

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        self._cnn = cnn
        self._rnn_state_size = rnn_state_size
    # pylint: enable=too-many-arguments

    @tensor
    def bidirectional_rnn(self) -> Tuple[Tuple[tf.Tensor, tf.Tensor],
                                         Tuple[tf.Tensor, tf.Tensor]]:
        return tf.nn.bidirectional_dynamic_rnn(
            OrthoGRUCell(self._rnn_state_size),
            OrthoGRUCell(self._rnn_state_size),
            self.cnn_temporal_states,
            sequence_length=self.cnn_lengths,
            dtype=tf.float32)

    @tensor
    def states(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[0], 2)
        # pylint: enable=unsubscriptable-object

    @tensor
    def temporal_states(self) -> tf.Tensor:
        return self.states

    @tensor
    def output(self) -> tf.Tensor:
        # pylint: disable=unsubscriptable-object
        return tf.concat(self.bidirectional_rnn[1], 1)
        # pylint: enable=unsubscriptable-object

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        return self.cnn_temporal_mask

    @tensor
    def cnn_temporal_states(self):
        states = tf.transpose(self._cnn.spatial_states, perm=[0, 2, 1, 3])
        shape = states.get_shape()
        res = tf.reshape(
            states, [-1, shape[1].value, shape[2].value * shape[3].value])
        return res

    @tensor
    def cnn_temporal_mask(self) -> tf.Tensor:
        mask = tf.squeeze(self._cnn.spatial_mask, 3)
        summed = tf.reduce_sum(mask, axis=1)
        return tf.to_float(tf.greater(summed, 0))

    @tensor
    def cnn_lengths(self):
        return tf.to_int32(tf.reduce_sum(self.cnn_temporal_mask, 1))

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {}

    def get_dependencies(self) -> Set["ModelPart"]:
        """Collect recusively all encoders and decoders."""
        return self._cnn.get_dependencies().union([self])
