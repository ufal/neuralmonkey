import math
from typing import Tuple

import tensorflow as tf

from neuralmonkey.nn.projection import linear


class NoisyGRUCell(tf.contrib.rnn.RNNCell):
    """
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with noisy
    activation functions (http://arxiv.org/abs/1603.00391). The theano code is
    availble at https://github.com/caglar/noisy_units.

    It is based on the TensorFlow implementatin of GRU just the activation
    function are changed for the noisy ones.
    """

    def __init__(self, num_units: int, training) -> None:
        self._num_units = num_units
        self.training = training

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state,
                 scope=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(linear([inputs, state], 2 * self._num_units),
                                2, 1)
                r, u = noisy_sigmoid(
                    r, self.training), noisy_sigmoid(u, self.training)
        with tf.variable_scope("Candidate"):
            c = noisy_tanh(linear([inputs, r * state], self._num_units),
                           self.training)
            new_h = u * state + (1 - u) * c
        return new_h, new_h


def noisy_activation(x, generic, linearized, training,
                     alpha: float=1.1, c: float=0.5):
    """
    Implements the noisy activation with Half-Normal Noise for Hard-Saturation
    functions. See http://arxiv.org/abs/1603.00391, Algorithm 1.

    Args:

        x: Tensor which is an input to the activation function

        generic: The generic formulation of the activation function. (denoted
            as h in the paper)

        linearized: Linearization of the activation based on the first-order
            Tailor expansion around zero. (denoted as u in the paper)

        training: A boolean tensor telling whether we are in the training stage
            (and the noise is sampled) or in runtime when the expactation is
            used instead.

        alpha: Mixing hyper-parameter. The leakage rate from the linearized
            function to the nonlinear one.

        c: Standard deviation of the sampled noise.

    """

    delta = generic(x) - linearized(x)
    d = -tf.sign(x) * tf.sign(1 - alpha)
    p = tf.get_variable("p", shape=[1], initializer=tf.ones_initializer())
    scale = c * (tf.sigmoid(p * delta) - 0.5) ** 2
    noise = tf.where(training, tf.abs(
        tf.random_normal([])), math.sqrt(2 / math.pi))
    activation = alpha * generic(x) + (1 - alpha) * \
        linearized(x) + d * scale * noise
    return activation


# lin_sigmoid, hard_tanh, hard_sigmoid are equations (1), (3) and (4) in
# the Noisy Activation Functions paper

def noisy_sigmoid(x, training):
    def lin_sigmoid(x):
        return 0.25 * x + 0.5

    def hard_sigmoid(x):
        return tf.minimum(tf.maximum(lin_sigmoid(x), 0.), 1.)
    return noisy_activation(x, hard_sigmoid, lin_sigmoid, training)


def noisy_tanh(x, training):
    def hard_tanh(x):
        return tf.minimum(tf.maximum(x, -1.), 1.)
    return noisy_activation(x, hard_tanh, lambda y: y, training)
