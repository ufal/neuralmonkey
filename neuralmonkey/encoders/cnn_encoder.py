"""CNN for image processing."""

from typing import List, Tuple, Type, Optional
from typeguard import check_argument_types

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
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell


class CNNEncoder(ModelPart, Attentive):
    """An image encoder.

    It projects the input image through a sere of convolutioal operations. The
    projected image is vertically cut and fed to stacked RNN layers which
    encode the image into a single vector.

    Attributes:
        input_op: Placeholder for the batch of input images
        padding_masks: Placeholder for matrices capturing telling where the
            image has been padded.
        image_processing_layers: List of TensorFlow operator that are
            visualize able image transformations.
        encoded: Operator that returns a batch of encoded image (intended
            as an input for the decoder).
        attention_tensor: Tensor computing a batch of attention
            matrices for the decoder.
        train_mode: Placeholder for boolean telling whether the training
            is running.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self,
                 name: str,
                 data_id: str,
                 convolutions: List[Tuple[int, int, Optional[int]]],
                 image_height: int, image_width: int, pixel_dim: int,
                 fully_connected: Optional[List[int]]=None,
                 batch_normalization: bool=True,
                 local_response_normalization: bool=True,
                 dropout_keep_prob: float=0.5,
                 attention_type: Type=Attention,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
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
        assert check_argument_types()
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(self, attention_type)

        self.data_id = data_id
        self.dropout_keep_prob = dropout_keep_prob

        with tf.variable_scope(name) as self._scope:
            self.dropout_placeholder = tf.placeholder(
                tf.float32, name="dropout")
            self.train_mode = tf.placeholder(tf.bool, shape=[],
                                             name="mode_placeholder")
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
            self.last_padding_masks = last_padding_masks

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
                    dropout_plc=self.dropout_placeholder)

            self._cnn_attention_tensor = tf.reshape(
                last_layer, [-1, last_width * last_height, last_n_channels])

            self._cnn_attention_mask = tf.reshape(
                last_padding_masks, [-1, last_width * last_height])

    @property
    def _attention_tensor(self) -> tf.Tensor:
        return self._cnn_attention_tensor

    @property
    def _attention_mask(self) -> tf.Tensor:
        return self._cnn_attention_mask

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        # if it is from the pickled file, it is list, not numpy tensor,
        # so convert it as as a prevention
        images = np.array(dataset.get_series(self.data_id))

        f_dict = {}
        f_dict[self.input_op] = images / 225.0

        # it is one everywhere where non-zero, i.e. zero columns are masked out
        f_dict[self.padding_masks] = np.sign(
            np.sum(images, axis=3, keepdims=True))
        assert np.all(f_dict[self.padding_masks] * images == images)

        if train:
            f_dict[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            f_dict[self.dropout_placeholder] = 1.0
        f_dict[self.train_mode] = train
        return f_dict


class CNNRNNEncoder(CNNEncoder):
    """A decoder with RNN layer on top of a CNN.

    Typical use of this encoder is in OCR where the image is first processed
    with a convolution and then horizontally sliced and encoded using an RNN.
    """
    # pylint: disable=too-many-locals,too-many-arguments
    def __init__(self,
                 name: str,
                 data_id: str,
                 convolutions: List[Tuple[int, int, Optional[int]]],
                 image_height: int, image_width: int, pixel_dim: int,
                 rnn_size: int,
                 batch_normalization: bool=True,
                 local_response_normalization: bool=True,
                 dropout_keep_prob: float=0.5,
                 attention_type: Type=Attention,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        assert check_argument_types()
        CNNEncoder.__init__(self, name, data_id, convolutions,
                            image_height, image_width, pixel_dim, None,
                            batch_normalization, local_response_normalization,
                            dropout_keep_prob, attention_type, save_checkpoint,
                            load_checkpoint)

        last_layer = self.image_processing_layers[-1]
        last_height, last_width, last_n_channels = [
            s.value for s in last_layer.get_shape()[1:]]

        with tf.variable_scope(self._scope):
            with tf.variable_scope("post_cnn_rnn"):
                rnn_input = tf.reshape(
                    tf.transpose(last_layer, [0, 2, 1, 3]),
                    [-1, last_width, last_height * last_n_channels])
                rnn_mask = tf.sign(
                    tf.reduce_sum(self.last_padding_masks, [1, 2]))
                rnn_lengths = tf.to_int32(tf.reduce_sum(rnn_mask, [1]))

                (outputs_bidi_tup,
                 encoded_tup) = tf.nn.bidirectional_dynamic_rnn(
                     OrthoGRUCell(rnn_size), OrthoGRUCell(rnn_size),
                     rnn_input, sequence_length=rnn_lengths,
                     dtype=tf.float32)

                hidden_states = tf.concat(2, outputs_bidi_tup)

                with tf.variable_scope('attention_tensor'):
                    self._cnn_attention_tensor = dropout(
                        hidden_states, dropout_keep_prob, self.train_mode)
                    self._cnn_attention_mask = rnn_mask

                self.encoded = tf.concat(1, encoded_tup)
    # pylint: enable=too-many-locals,too-many-arguments
