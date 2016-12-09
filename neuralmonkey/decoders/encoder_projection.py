"""
This module contains different variants of projection of encoders into the
initial state of the decoder.
"""
#tests: mypy, lint
from typing import List, Optional, Callable

import tensorflow as tf

from neuralmonkey.nn.utils import dropout
from neuralmonkey.nn.projection import linear
from neuralmonkey.logging import log

#pylint: disable=unused-argument
# The function must conform the API
def empty_initial_state(train_mode: tf.Tensor,
                        rnn_size: Optional[int],
                        encoders: Optional[List[object]]=None) -> tf.Tensor:
    """Return an empty vector

    Arguments:
        train_mode: tf 0-D bool Tensor specifying the training mode (not used)
        rnn_size: The size of the resulting vector
        encoders: The list of encoders (not used)
    """
    if rnn_size is None:
        raise ValueError("You must supply rnn_size for this type of "
                         "encoder projection")
    return tf.zeros([rnn_size])
#pylint: enable=unused-argument


def linear_encoder_projection(
        dropout_keep_prob: float) -> Callable[
            [tf.Tensor, Optional[int], Optional[List[object]]],
            tf.Tensor]:
    """Return a projection function which applies dropout on concatenated
    encoder final states and returns a linear projection to a rnn_size-sized
    tensor.

    Arguments:
        dropout_keep_prob: The dropout keep probability
    """
    def func(train_mode: tf.Tensor,
             rnn_size: Optional[int]=None,
             encoders: Optional[List[object]]=None) -> tf.Tensor:
        """Linearly project the encoders' encoded value to rnn_size
        and apply dropout

        Arguments:
            train_mode: tf 0-D bool Tensor specifying the training mode
            rnn_size: The size of the resulting vector
            encoders: The list of encoders
        """
        if rnn_size is None:
            raise ValueError("You must supply rnn_size for this type of "
                             "encoder projection")

        if encoders is None or not encoders:
            raise ValueError("There must be at least one encoder for this type "
                             "of encoder projection")

        with tf.variable_scope("encoders_projection") as scope:
            encoded_concat = tf.concat(1, [e.encoded for e in encoders])
            encoded_concat = dropout(
                encoded_concat, dropout_keep_prob, train_mode)

            return linear(encoded_concat, rnn_size, scope)

    return func


def concat_encoder_projection(
        train_mode: tf.Tensor,
        rnn_size: Optional[int]=None,
        encoders: Optional[List[object]]=None) -> tf.Tensor:
    """Create the initial state by concatenating the encoders' encoded values

    Arguments:
        train_mode: tf 0-D bool Tensor specifying the training mode (not used)
        rnn_size: The size of the resulting vector (not used)
        encoders: The list of encoders
    """
    if encoders is None or not encoders:
        raise ValueError("There must be at least one encoder for this type "
                         "of encoder projection")

    if rnn_size is not None:
        log("Warning: The rnn_size argument is ignored in this type of "
            "encoder projection", color="red")

    encoded_concat = tf.concat(1, [e.encoded for e in encoders])

    log("The inferred rnn_size of this encoder projection will be {}"
        .format(encoded_concat.get_shape()[1].value))

    return encoded_concat
