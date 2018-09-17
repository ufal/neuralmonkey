"""Encoder Projection Module.

This module contains different variants of projection of encoders into the
initial state of the decoder.

Encoder projections are specified in the configuration file.  Each encoder
projection function has a unified type ``EncoderProjection``, which is a
callable that takes three arguments:

1. ``train_mode`` -- boolean tensor specifying whether the train mode is on
2. ``rnn_size`` -- the size of the resulting initial state
3. ``encoders`` -- a list of ``Stateful`` objects used as the encoders.

To enable further parameterization of encoder projection functions, one can
use higher-order functions.
"""
from typing import List, Callable, cast

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.model.stateful import Stateful, TemporalStatefulWithOutput
from neuralmonkey.nn.utils import dropout
from neuralmonkey.nn.ortho_gru_cell import orthogonal_initializer
from neuralmonkey.logging import log
from neuralmonkey.tf_utils import get_initializer


# pylint: disable=invalid-name
EncoderProjection = Callable[
    [tf.Tensor, int, List[Stateful]], tf.Tensor]
# pylint: enable=invalid-name


# pylint: disable=unused-argument
# The function must conform the API
def empty_initial_state(train_mode: tf.Tensor,
                        rnn_size: int,
                        encoders: List[Stateful] = None) -> tf.Tensor:
    """Return an empty vector."""
    if rnn_size is None:
        raise ValueError(
            "You must supply rnn_size for this type of encoder projection")
    return tf.zeros([rnn_size])


def linear_encoder_projection(dropout_keep_prob: float) -> EncoderProjection:
    """Return a linear encoder projection.

    Return a projection function which applies dropout on concatenated
    encoder final states and returns a linear projection to a rnn_size-sized
    tensor.

    Arguments:
        dropout_keep_prob: The dropout keep probability
    """
    check_argument_types()

    def func(train_mode: tf.Tensor,
             rnn_size: int,
             encoders: List[Stateful]) -> tf.Tensor:

        if rnn_size is None:
            raise ValueError(
                "You must supply rnn_size for this type of encoder projection")

        en_concat = concat_encoder_projection(train_mode, None, encoders)
        en_concat = dropout(en_concat, dropout_keep_prob, train_mode)

        return tf.layers.dense(en_concat, rnn_size, name="encoders_projection")

    return cast(EncoderProjection, func)


def concat_encoder_projection(
        train_mode: tf.Tensor,
        rnn_size: int = None,
        encoders: List[Stateful] = None) -> tf.Tensor:
    """Concatenate the encoded values of the encoders."""

    if encoders is None or not encoders:
        raise ValueError("There must be at least one encoder for this type "
                         "of encoder projection")

    output_size = sum(e.output.get_shape()[1].value for e in encoders)
    if rnn_size is not None and rnn_size != output_size:
        raise ValueError("RNN size supplied for concat projection ({}) does "
                         "not match the size of the concatenated vectors ({})."
                         .format(rnn_size, output_size))

    log("The inferred rnn_size of this encoder projection will be {}"
        .format(output_size))

    encoded_concat = tf.concat([e.output for e in encoders], 1)
    return encoded_concat


def nematus_projection(dropout_keep_prob: float = 1.0) -> EncoderProjection:
    """Return encoder projection used in Nematus.

    The initial state is a dense projection with tanh activation computed on
    the averaged states of the encoders. Dropout is applied to the means
    (before the projection).

    Arguments:
        dropout_keep_prob: The dropout keep probability.
    """
    check_argument_types()

    def func(
            train_mode: tf.Tensor,
            rnn_size: int,
            encoders: List[TemporalStatefulWithOutput]) -> tf.Tensor:

        if len(encoders) != 1:
            raise ValueError("Exactly one encoder required for this type of "
                             "projection. {} given.".format(len(encoders)))
        encoder = encoders[0]

        # shape (batch, time)
        masked_sum = tf.reduce_sum(
            encoder.temporal_states
            * tf.expand_dims(encoder.temporal_mask, 2), 1)

        # shape (batch, 1)
        lengths = tf.reduce_sum(encoder.temporal_mask, 1, keepdims=True)

        means = masked_sum / lengths
        means = dropout(means, dropout_keep_prob, train_mode)

        encoder_rnn_size = means.get_shape()[1].value

        kernel_initializer = orthogonal_initializer()
        if encoder_rnn_size != rnn_size:
            kernel_initializer = None

        return tf.layers.dense(
            means, rnn_size, activation=tf.tanh,
            kernel_initializer=get_initializer(
                "encoders_projection/kernel", kernel_initializer),
            name="encoders_projection")

    return cast(EncoderProjection, func)
