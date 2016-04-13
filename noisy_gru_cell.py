import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import RNNCell
import math

class NoisyGRUCell(RNNCell):
    """
    Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with noisy
    activation functions (http://arxiv.org/abs/1603.00391). The theano code is
    availble at https://github.com/caglar/noisy_units.

    It is based on the TensorFlow implementatin of GRU just the activation
    function are changed for the noisy ones.
    """
    def __init__(self, num_units, training, input_size=None, batch_normalize=False):
        """
        Instantiates the recurrent unit.

        Args:

            num_units: Number of hidden units.

            training: Placeholder telling whether we are in the training or
                inference time.

            input_size: Size of the input (if it is different from the nuber of
                huidden units).

            batch_norm: Flag whther the batch normalizatigon should be used.

        """


        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self.training = training
        self.batch_norm = batch_normalize

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):    # "GRUCell"
            with tf.variable_scope("Gates"):    # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.

                r = linear([inputs, state], self._num_units, True, 1.0, "remember", batch_normalize=self.batch_norm, is_training=self.training)
                u = linear([inputs, state], self._num_units, True, 1.0, "update", batch_normalize=self.batch_norm, is_training=self.training)

                r, u = noisy_sigmoid(r, self.training), noisy_sigmoid(u, self.training)

            with tf.variable_scope("Candidate"):
                c = noisy_tanh(linear([inputs, r * state], self._num_units, True), self.training)
                if self.batch_norm:
                    c = batch_norm(c, self._num_units, self.training)
            new_h = u * state + (1 - u) * c

        return new_h, new_h


def noisy_activation(x, generic, linearized, training, alpha=1.1, c=0.5):
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
    p = tf.Variable(1.0)
    scale = c * (tf.sigmoid(p * delta) - 0.5)  ** 2
    noise = tf.select(training, tf.abs(tf.random_normal([])), math.sqrt(2 / math.pi))
    activation = alpha * generic(x) + (1 - alpha) * linearized(x) + d * scale * noise
    return activation


# These are equations (1), (3) and (4) in the Noisy Activation FUnctions paper
lin_sigmoid = lambda x: 0.25 * x + 0.5
hard_tanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)
hard_sigmoid = lambda x: tf.minimum(tf.maximum(lin_sigmoid(x), 0.), 1.)


def noisy_sigmoid(x, training):
    return noisy_activation(x, hard_sigmoid, lin_sigmoid, training)


def noisy_tanh(x, training):
    return noisy_activation(x, hard_tanh, lambda y: y, training)


def batch_norm(x, n_out, phase_train, scope='batch_norm', affine=True):
    """
    Batch normalization for fully connected layers.
    Taken from http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    and modified to work with the fully connected layers.

    Arguments:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs

    Return:
        normed:      batch-normalized maps

    """

    with tf.variable_scope(scope):
        reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(reshaped_x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
            mean_var_with_update,
            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
            beta, gamma, 1e-3, affine)
        return tf.reshape(normed, shape=[-1, n_out])


def linear(args, output_size, bias, bias_start=0.0, scope=None, batch_normalize=False, is_training=None):
    with tf.variable_scope("linear" if scope is None else scope):
        projected = []
        for i, arg in enumerate(args):
            input_size = arg.get_shape()[1].value
            W_i = tf.get_variable(shape=[input_size, output_size], name="proj_{}".format(i))
            proj = tf.matmul(arg, W_i)
            if batch_normalize:
                proj = batch_norm(proj, output_size, is_training)
            projected.append(proj)

        res = sum(projected)

        bias_term = tf.get_variable(
                "Bias", [output_size],
                initializer=tf.constant_initializer(bias_start))
        return res + bias_term
