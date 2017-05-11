"""
This module provides utility functions used across the package.
"""
import tensorflow as tf


def dropout(variable: tf.Tensor,
            keep_prob: float,
            train_mode: tf.Tensor) -> tf.Tensor:
    """Performs dropout on a variable, depending on mode.

    Arguments:
        variable: The variable to be dropped out
        keep_prob: The probability of keeping a value in the variable
        train_mode: A bool Tensor specifying whether to dropout or not
    """
    # Maintain clean graph - no dropout op when there is none applied
    if keep_prob == 1.0:
        return variable

    dropped_value = tf.nn.dropout(variable, keep_prob)
    return tf.where(train_mode, dropped_value, variable)
