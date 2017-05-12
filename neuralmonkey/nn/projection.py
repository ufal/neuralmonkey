"""
This module implements various types of projections.
"""
import tensorflow as tf
from neuralmonkey.nn.utils import dropout


def linear(inputs, size, scope="LinearProjection"):
    """Simple linear projection

    y = Wx + b

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors with
                equal length in the first dimension (batch size)
        size: The size of dimension 1 of the output tensor.
        scope: The name of the scope used for the variables.

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope):
        if isinstance(inputs, list):
            # if there is a list of tensor on the input, concatenate along
            # the last dimension and project.
            inputs = tf.concat(inputs, axis=-1)

        return tf.contrib.layers.fully_connected(
            inputs, size, biases_initializer=tf.zeros_initializer(),
            activation_fn=None, scope=scope)


def nonlinear(inputs, size, activation, scope="NonlinearProjection"):
    """Linear projection with non-linear activation function

    y = activation(Wx + b)

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors
                with equal length in the first dimension (batch size)
        size: The size of the second dimension (index 1) of the output tensor
        scope: The name of the scope used for the variables

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope) as varscope:
        return activation(linear(inputs, size, scope=varscope))


def maxout(inputs, size, scope="MaxoutProjection"):
    """Implementation of Maxout layer (Goodfellow et al., 2013)
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
        projected = linear(inputs, size * 2, scope=scope)
        maxout_input = tf.reshape(projected, [-1, 1, 2, size])
        maxpooled = tf.nn.max_pool(
            maxout_input, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

        reshaped = tf.reshape(maxpooled, [-1, size])
        return reshaped


def multilayer_projection(input_, layer_sizes, train_mode: tf.Tensor,
                          activation=tf.nn.relu, dropout_keep_prob=1.0,
                          scope="mlp"):
    mlp_input = input_

    with tf.variable_scope(scope):
        for i, size in enumerate(layer_sizes):
            mlp_input = nonlinear(mlp_input, size, activation=activation,
                                  scope="mlp_layer_{}".format(i))
            mlp_input = dropout(mlp_input, dropout_keep_prob, train_mode)

    return mlp_input
