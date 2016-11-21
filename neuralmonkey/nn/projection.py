"""
This module implements various types of projections.
"""
#tests: lint
import tensorflow as tf

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
        return tf.nn.seq2seq.linear(inputs, size, True)

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
        projected = linear(inputs, size * 2)
        maxout_input = tf.reshape(projected, [-1, 1, 2, size])
        maxpooled = tf.nn.max_pool(
            maxout_input, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

        reshaped = tf.reshape(maxpooled, [-1, size])
        return reshaped
