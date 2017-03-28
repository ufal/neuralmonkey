"""
This module implements the highway networks.
"""
import tensorflow as tf


def highway(inputs, activation=tf.nn.relu, scope="HighwayNetwork"):
    """Simple highway layer

    y = H(x, Wh) * T(x, Wt) + x * C(x, Wc)

    where:

    C(x, Wc) = 1 - T(x, Wt)

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors with
                equal length in the first dimension (batch size)
        activation: Activation function of the linear part of the formula
                H(x, Wh).
        scope: The name of the scope used for the variables.

    Returns:
        A tensor of shape tf.shape(inputs)
    """
    with tf.variable_scope(scope):
        if isinstance(inputs, list):
            # if there is a list of tensor on the input, concatenate along
            # the last dimension and project.
            inputs = tf.concat(inputs, axis=-1)

        # pylint: disable=no-member
        vec_size = inputs.get_shape().as_list()[-1]

        # pylint: disable=invalid-name
        W_shape = [vec_size, vec_size]
        b_shape = [vec_size]

        W_H = tf.get_variable(
            "weight_H",
            shape=W_shape,
            initializer=tf.random_normal_initializer(stddev=0.1))
        b_H = tf.get_variable(
            "bias_H",
            shape=b_shape,
            initializer=tf.constant_initializer(-1.0))

        W_T = tf.get_variable(
            "weight_T",
            shape=W_shape,
            initializer=tf.random_normal_initializer(stddev=0.1))
        b_T = tf.get_variable(
            "bias_T",
            shape=b_shape,
            initializer=tf.constant_initializer(-1.0))

        T = tf.sigmoid(
            tf.add(tf.matmul(inputs, W_T), b_T),
            name="transform_gate")
        H = activation(
            tf.add(tf.matmul(inputs, W_H), b_H),
            name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(
            tf.multiply(H, T),
            tf.multiply(inputs, C),
            "y")
        return y
