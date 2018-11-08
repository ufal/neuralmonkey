"""CNN for image processing."""

from typing import cast, Callable, List, Tuple, Set, Union
from typeguard import check_argument_types

import numpy as np
import tensorflow as tf

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import (
    ModelPart, GenericModelPart, FeedDict, InitializerSpecs)
from neuralmonkey.model.stateful import (SpatialStatefulWithOutput,
                                         TemporalStatefulWithOutput)
from neuralmonkey.nn.projection import multilayer_projection


# Tuples used for configuration of the convolutional layers. See docstring of
# CNNEncoder initialization for more details.
# pylint: disable=invalid-name
ConvSpec = Tuple[str, int, int, str, int]
ResNetSpec = Tuple[str, int, int]
MaxPoolSpec = Tuple[str, int, int, str]
# pylint: enable=invalid-name


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
                 convolutions: List[Union[ConvSpec, ResNetSpec, MaxPoolSpec]],
                 image_height: int, image_width: int, pixel_dim: int,
                 fully_connected: List[int] = None,
                 batch_normalize: bool = False,
                 dropout_keep_prob: float = 0.5,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize a convolutional network for image processing.

        The convolutional network can consist of plain convolutions,
        max-pooling layers and residual block. In the configuration, they are
        specified using the following tuples.

            * convolution: ("C", kernel_size, stride, padding, out_channel);
            * max / average pooling: ("M"/"A", kernel_size, stride, padding);
            * residual block: ("R", kernel_size, out_channels).

        Padding must be either "valid" or "same".

        Args:
            convolutions: Configuration of convolutional layers.
            data_id: Identifier of the data series in the dataset.
            image_height: Height of the input image in pixels.
            image_width: Width of the image.
            pixel_dim: Number of color channels in the input images.
            dropout_keep_prob: Probability of keeping neurons active in
                dropout. Dropout is done between all convolutional layers and
                fully connected layer.
        """
        check_argument_types()
        ModelPart.__init__(
            self, name, reuse, save_checkpoint, load_checkpoint, initializers)

        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        self.image_height = image_height
        self.image_width = image_width
        self.pixel_dim = pixel_dim
        self.convolutions = convolutions
        self.fully_connected = fully_connected
        self.batch_normalize = batch_normalize
    # pylint: enable=too-many-arguments, too-many-locals

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

    def batch_norm_callback(self, layer_output: tf.Tensor) -> tf.Tensor:
        if self.batch_normalize:
            return tf.layers.batch_normalization(
                layer_output, training=self.train_mode)
        return layer_output

    @tensor
    def image_processing_layers(self) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        """Do all convolutions and return the last conditional map.

        No dropout is applied between the convolutional layers. By default, the
        activation function is ReLU.
        """
        last_layer = self.image_input
        last_mask = self.image_mask
        last_channels = self.pixel_dim
        image_processing_layers = []  # type: List[Tuple[tf.Tensor, tf.Tensor]]

        with tf.variable_scope("convolutions"):
            for i, specification in enumerate(self.convolutions):
                if specification[0] == "C":
                    (last_layer, last_mask,
                     last_channels) = plain_convolution(
                         last_layer, last_mask,
                         cast(ConvSpec, specification),
                         self.batch_norm_callback, i)
                    image_processing_layers.append((last_layer, last_mask))
                elif specification[0] in ["M", "A"]:
                    last_layer, last_mask = pooling(
                        last_layer, last_mask,
                        cast(MaxPoolSpec, specification), i)
                    image_processing_layers.append((last_layer, last_mask))
                elif specification[0] == "R":
                    if not self.batch_normalize:
                        raise ValueError(
                            "Using ResNet blocks requires batch normalization "
                            "to be turned on.")
                    (last_layer, last_mask,
                     last_channels) = residual_block(
                         last_layer, last_mask, last_channels,
                         cast(ResNetSpec, specification),
                         self.batch_norm_callback, i)
                    image_processing_layers.append((last_layer, last_mask))
                else:
                    raise ValueError(
                        "Unknown type of convoutional layer #{}: '{}'".format(
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
        fd = ModelPart.feed_dict(self, dataset, train)

        # if it is from the pickled file, it is a list, not a numpy tensor,
        # so convert it as as a prevention
        images = np.array(list(dataset.get_series(self.data_id)))

        fd[self.image_input] = images / 255.0

        # the image mask is one everywhere where the image is non-zero, i.e.
        # zero pixels are masked out
        fd[self.image_mask] = np.sign(np.sum(images, axis=3, keepdims=True))

        return fd


def plain_convolution(
        prev_layer: tf.Tensor,
        prev_mask: tf.Tensor,
        specification: ConvSpec,
        batch_norm_callback: Callable[[tf.Tensor], tf.Tensor],
        layer_num: int) -> Tuple[tf.Tensor, tf.Tensor, int]:
    try:
        check_argument_types()
    except TypeError as err:
        raise ValueError((
            "Specification of a convolutional layer (number {} in config) "
            'needs to have 5 members: "C", kernel size, stride, '
            "padding, output channels, was {}").format(
                layer_num, specification)) from err
    kernel_size, stride, pad, out_channels = specification[1:]

    if pad not in ["same", "valid"]:
        raise ValueError(
            ("Padding must be 'same' or 'valid', "
             "was '{}' in layer {}.").format(pad, layer_num + 1))

    with tf.variable_scope("layer_{}_convolution".format(layer_num)):
        next_layer = tf.layers.conv2d(
            prev_layer, out_channels, kernel_size,
            activation=None, padding=pad)

        next_layer = batch_norm_callback(next_layer)
        next_layer = tf.nn.relu(next_layer)

        next_mask = tf.layers.max_pooling2d(
            prev_mask, kernel_size, stride, padding=pad)

    return next_layer, next_mask, out_channels


def residual_block(
        prev_layer: tf.Tensor,
        prev_mask: tf.Tensor,
        prev_channels: int,
        specification: ResNetSpec,
        batch_norm_callback: Callable[[tf.Tensor], tf.Tensor],
        layer_num: int) -> Tuple[tf.Tensor, tf.Tensor, int]:
    try:
        check_argument_types()
    except TypeError as err:
        raise ValueError((
            "Specification of a residual block (number {} in config) "
            'needs to have 3 members: "R", kernel size, channels; '
            "was {}").format(layer_num, specification)) from err
    kernel_size, out_channels = specification[1:]

    with tf.variable_scope("layer_{}_resnet_block".format(layer_num)):
        if out_channels == prev_channels:
            before_resnet_block = prev_layer
        else:
            with tf.variable_scope("project_input"):
                before_resnet_block = tf.layers.conv2d(
                    prev_layer, out_channels, 1, 1,
                    "same", activation=None)
                before_resnet_block = batch_norm_callback(before_resnet_block)

        with tf.variable_scope("conv_a"):
            after_cnn = batch_norm_callback(prev_layer)
            after_cnn = tf.nn.relu(after_cnn)
            after_cnn = tf.layers.conv2d(
                after_cnn, out_channels, kernel_size,
                padding="same", activation=None)

        with tf.variable_scope("conv_b"):
            after_cnn = batch_norm_callback(after_cnn)
            after_cnn = tf.nn.relu(after_cnn)
            after_cnn = tf.layers.conv2d(
                after_cnn, out_channels, kernel_size,
                padding="same", activation=None)

        next_layer = after_cnn + before_resnet_block

    return next_layer, prev_mask, out_channels


def pooling(
        prev_layer: tf.Tensor,
        prev_mask: tf.Tensor,
        specification: MaxPoolSpec,
        layer_num: int) -> Tuple[tf.Tensor, tf.Tensor]:
    try:
        check_argument_types()
    except TypeError as err:
        raise ValueError((
            "Specification of a max-pooling layer (number {} in config) "
            'needs to have 3 members: "M", pool size, stride, padding, '
            "was {}").format(layer_num, specification)) from err
    pool_type, pool_size, stride, pad = specification

    if pool_type == "M":
        pool_fn = tf.layers.max_pooling2d
    elif pool_type == "A":
        pool_fn = tf.layers.average_pooling2d
    else:
        raise ValueError(
            ("Unsupported type of pooling: {}, use 'M' for max-pooling or "
             "'A' for average pooling.").format(pool_type))

    if pad not in ["same", "valid"]:
        raise ValueError(
            "Padding must be 'same' or 'valid', was '{}' in layer {}."
            .format(pad, layer_num + 1))

    with tf.variable_scope("layer_{}_max_pool".format(layer_num)):
        next_layer = pool_fn(prev_layer, pool_size, stride)
        next_mask = tf.layers.max_pooling2d(prev_mask, pool_size, stride)
    return next_layer, next_mask


class CNNTemporalView(ModelPart, TemporalStatefulWithOutput):
    """Slice the convolutional maps left to right."""

    def __init__(self,
                 name: str,
                 cnn: CNNEncoder) -> None:
        check_argument_types()
        ModelPart.__init__(
            self, name, save_checkpoint=None, load_checkpoint=None)
        self._cnn = cnn

    @tensor
    def output(self) -> tf.Tensor:
        return self._cnn.output

    @tensor
    def temporal_states(self):
        states = tf.transpose(self._cnn.spatial_states, perm=[0, 2, 1, 3])
        shape = states.get_shape()
        res = tf.reshape(
            states, [-1, shape[1].value, shape[2].value * shape[3].value])
        return res

    @tensor
    def temporal_mask(self) -> tf.Tensor:
        mask = tf.squeeze(self._cnn.spatial_mask, 3)
        summed = tf.reduce_sum(mask, axis=1)
        return tf.to_float(tf.greater(summed, 0))

    def get_dependencies(self) -> Set[GenericModelPart]:
        """Collect recusively all encoders and decoders."""
        return self._cnn.get_dependencies().union([self])
