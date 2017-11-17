"""Encoder Projection Module.

This module contains different variants of projection of encoders into the
initial state of the decoder.
"""
from typing import List, Callable, cast

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import Stateful, TemporalStatefulWithOutput
from neuralmonkey.nn.utils import dropout
from neuralmonkey.logging import log


# pylint: disable=invalid-name
EncoderProjection = Callable[
    [tf.Tensor, int, List[Stateful]], tf.Tensor]
# pylint: enable=invalid-name


# pylint: disable=unused-argument
# The function must conform the API
def empty_initial_state(train_mode: tf.Tensor,
                        rnn_size: int,
                        encoders: List[Stateful] = None) -> tf.Tensor:
    """Return an empty vector.

    Arguments:
        train_mode: tf 0-D bool Tensor specifying the training mode (not used)
        rnn_size: The size of the resulting vector
        encoders: The list of encoders (not used)
    """
    if rnn_size is None:
        raise ValueError("You must supply rnn_size for this type of "
                         "encoder projection")
    return tf.zeros([rnn_size])


def linear_encoder_projection(dropout_keep_prob: float) -> EncoderProjection:
    """Return a linear encoder projection.

    Return a projection function which applies dropout on concatenated
    encoder final states and returns a linear projection to a rnn_size-sized
    tensor.

    Arguments:
        dropout_keep_prob: The dropout keep probability
    """
    def func(train_mode: tf.Tensor,
             rnn_size: int,
             encoders: List[Stateful]) -> tf.Tensor:
        """Linearly project encoders' encoded value.

        Linearly project encoders' encoded value to rnn_size
        and apply dropout.

        Arguments:
            train_mode: tf 0-D bool Tensor specifying the training mode
            rnn_size: The size of the resulting vector
            encoders: The list of encoders
        """
        if rnn_size is None:
            raise ValueError("You must supply rnn_size for this type of "
                             "encoder projection")

        if not encoders:
            raise ValueError("There must be at least one encoder for this type"
                             " of encoder projection")

        encoded_concat = tf.concat([e.output for e in encoders], 1)
        encoded_concat = dropout(
            encoded_concat, dropout_keep_prob, train_mode)

        return tf.layers.dense(encoded_concat, rnn_size,
                               name="encoders_projection")

    return func


def concat_encoder_projection(
        train_mode: tf.Tensor,
        rnn_size: int = None,
        encoders: List[Stateful] = None) -> tf.Tensor:
    """Create the initial state by concatenating the encoders' encoded values.

    Arguments:
        train_mode: tf 0-D bool Tensor specifying the training mode (not used)
        rnn_size: The size of the resulting vector (not used)
        encoders: The list of encoders
    """
    if encoders is None or not encoders:
        raise ValueError("There must be at least one encoder for this type "
                         "of encoder projection")

    if rnn_size is not None:
        assert rnn_size == sum(e.output.get_shape()[1].value
                               for e in encoders)

    encoded_concat = tf.concat([e.output for e in encoders], 1)

    # pylint: disable=no-member
    log("The inferred rnn_size of this encoder projection will be {}"
        .format(encoded_concat.get_shape()[1].value))
    # pylint: enable=no-member

    return encoded_concat


def nematus_projection(dropout_keep_prob: float = 1.0) -> EncoderProjection:
    """Return a nematus-like encoder projection.

    Arguments:
        dropout_keep_prob: The dropout keep probability.
    """
    check_argument_types()

    def func(
            train_mode: tf.Tensor,
            rnn_size: int,
            encoders: List[TemporalStatefulWithOutput]) -> tf.Tensor:
        """Create the initial state by averaging the encoder's encoded values.

        This projection is used in Nematus models.

        Arguments:
            train_mode: tf 0-D bool Tensor specifying the training mode.
            rnn_size: The size of the resulting vector.
            encoders: The list of encoders. Must have length 1.
        """
        if len(encoders) != 1:
            raise ValueError("Exactly one encoder required for this type of "
                             "projection. {} given.".format(len(encoders)))
        encoder = encoders[0]

        # shape (batch, time)
        masked_sum = tf.reduce_sum(
            encoder.temporal_states
            * tf.expand_dims(encoder.temporal_mask, 2), 1)

        # shape (batch, 1)
        lengths = tf.reduce_sum(encoder.temporal_mask, 1, keep_dims=True)

        means = masked_sum / lengths
        means = dropout(means, dropout_keep_prob, train_mode)

        return tf.layers.dense(means, rnn_size,
                               activation=tf.tanh,
                               name="encoders_projection")

    return cast(EncoderProjection, func)
