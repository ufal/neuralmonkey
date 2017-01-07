"""
This module provides utility functions used across the package.
"""
# tests: mypy, lint
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
    # TODO maybe use math.isclose instead of this comparison
    if keep_prob == 1.0:
        return variable

    # TODO remove this line as soon as TF .12 is used.
    train_mode_selector = tf.fill(tf.shape(variable)[:1], train_mode)
    dropped_value = tf.nn.dropout(variable, keep_prob)
    return tf.select(train_mode_selector, dropped_value, variable)
