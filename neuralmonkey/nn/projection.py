"""Module which implements various types of projections."""
from typing import List, Callable
import tensorflow as tf
from neuralmonkey.nn.utils import dropout


def maxout(inputs: tf.Tensor,
           size: int,
           scope: str = "MaxoutProjection") -> tf.Tensor:
    """Apply a maxout operation.

    Implementation of Maxout layer (Goodfellow et al., 2013).

    http://arxiv.org/pdf/1302.4389.pdf

    z = Wx + b
    y_i = max(z_{2i-1}, z_{2i})

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors with
                equal length in the first dimension (batch size)
        size: The size of dimension 1 of the output tensor.
        scope: The name of the scope used for the variables

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope):
        projected = tf.layers.dense(inputs, size * 2, name=scope)
        maxout_input = tf.reshape(projected, [-1, 1, 2, size])
        maxpooled = tf.nn.max_pool(
            maxout_input, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

        reshaped = tf.reshape(maxpooled, [-1, size])
        return reshaped


def multilayer_projection(
        input_: tf.Tensor,
        layer_sizes: List[int],
        train_mode: tf.Tensor,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
        dropout_keep_prob: float = 1.0,
        scope: str = "mlp") -> tf.Tensor:
    mlp_input = input_

    with tf.variable_scope(scope):
        for i, size in enumerate(layer_sizes):
            mlp_input = tf.layers.dense(
                mlp_input,
                size,
                activation=activation,
                name="mlp_layer_{}".format(i))

            mlp_input = dropout(mlp_input, dropout_keep_prob, train_mode)

    return mlp_input


def glu(input_: tf.Tensor,
        gating_fn: Callable[[tf.Tensor], tf.Tensor] = tf.sigmoid) -> tf.Tensor:
    """Apply a Gated Linear Unit.

    Gated Linear Unit - Dauphin et al. (2016).

    http://arxiv.org/abs/1612.08083
    """
    dimensions = input_.get_shape().as_list()

    if dimensions[-1] % 2 != 0:
        raise ValueError("Input size should be an even number")

    lin, nonlin = tf.split(input_, 2, axis=len(dimensions) - 1)

    return lin * gating_fn(nonlin)
