"""Collection of learning rate decay strategies."""
from typing import Callable
import math

import tensorflow as tf

# pylint: disable=invalid-name
DecayFunction = Callable[[tf.Tensor], tf.Tensor]
# pylint: disable=invalid-name


def transformer_decay(model_dimension: int,
                      warmup_steps: int) -> DecayFunction:
    """Return decay function as defined in Vaswani et al., 2017, Equation 3.

    Arguments:
        model_dimension: Size of the hidden states of decoder and encoder
        warmup_steps: Number of warm-up steps
    """
    inv_sq_dim = 1 / math.sqrt(model_dimension)
    inv_sq3_warmup_steps = math.pow(warmup_steps, -1.5)

    def decay_function(step: tf.Tensor) -> tf.Tensor:
        inv_sq_step = 1 / tf.sqrt(tf.to_float(step))
        warmup = tf.to_float(step) * inv_sq3_warmup_steps
        return inv_sq_dim * tf.minimum(inv_sq_step, warmup)

    return decay_function


def constant_decay(decay_rate: float = 1.0) -> DecayFunction:
    """Return default decay function."""

    # pylint: disable=unused-argument
    def decay_function(step: tf.Tensor) -> tf.Tensor:
        return tf.constant(decay_rate)

    return decay_function
