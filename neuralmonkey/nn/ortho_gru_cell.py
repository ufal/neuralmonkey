import tensorflow as tf


def orthogonal_initializer():
    """Return an orthogonal initializer.

    Random orthogonal matrix is byproduct of singular value decomposition
    applied on a matrix initialized with normal distribution.

    The initializer works with 2D square matrices and matrices that can be
    splitted along axis 1 to several 2D matrices. In the latter case, each
    submatrix is initialized independently and the resulting orthogonal
    matrices are concatenated along axis 1.

    Note this is a higher order function in order to mimic the tensorflow
    initializer API.
    """

    # pylint: disable=unused-argument
    def func(shape, dtype, partition_info=None):
        if len(shape) != 2:
            raise ValueError(
                "Orthogonal initializer only works with 2D matrices.")

        if shape[1] % shape[0] != 0:
            raise ValueError("Shape {} is not compatible with orthogonal "
                             "initializer.".format(str(shape)))

        mult = int(shape[1] / shape[0])
        dim = shape[0]

        orthogonals = []
        for _ in range(mult):
            matrix = tf.random_normal([dim, dim], dtype=dtype)
            orthogonals.append(tf.svd(matrix)[1])

        return tf.concat(orthogonals, 1)
    # pylint: enable=unused-argument

    return func


# pylint: disable=too-few-public-methods
class OrthoGRUCell(tf.contrib.rnn.GRUCell):
    """Classic GRU cell but initialized using random orthogonal matrices."""

    def __init__(self, num_units, activation=None, reuse=None):
        tf.contrib.rnn.GRUCell.__init__(
            self, num_units, activation, reuse,
            kernel_initializer=tf.orthogonal_initializer())

    def __call__(self, inputs, state, scope="OrthoGRUCell"):
        return tf.contrib.rnn.GRUCell.__call__(self, inputs, state, scope)


# Note that tensorflow does not like when the type annotations are present.
class NematusGRUCell(tf.contrib.rnn.GRUCell):
    """Nematus implementation of gated recurrent unit cell.

    The main difference is the order in which the gating functions and linear
    projections are applied to the hidden state.

    The math is equivalent, in practice there are differences due to float
    precision errors.
    """

    def __init__(self, rnn_size, use_state_bias=False, use_input_bias=True):
        self.use_state_bias = use_state_bias
        self.use_input_bias = use_input_bias

        tf.contrib.rnn.GRUCell.__init__(self, rnn_size)

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope("gates"):
            input_to_gates = tf.layers.dense(
                inputs, 2 * self._num_units, name="input_proj",
                use_bias=self.use_input_bias)

            # Nematus does the orthogonal initialization probably differently
            state_to_gates = tf.layers.dense(
                state, 2 * self._num_units,
                use_bias=self.use_state_bias,
                kernel_initializer=orthogonal_initializer(),
                name="state_proj")

            gates_input = state_to_gates + input_to_gates
            reset, update = tf.split(
                tf.sigmoid(gates_input), num_or_size_splits=2, axis=1)

        with tf.variable_scope("candidate"):
            input_to_candidate = tf.layers.dense(
                inputs, self._num_units, use_bias=self.use_input_bias,
                name="input_proj")

            state_to_candidate = tf.layers.dense(
                state, self._num_units, use_bias=self.use_state_bias,
                kernel_initializer=orthogonal_initializer(),
                name="state_proj")

            candidate = self._activation(
                state_to_candidate * reset + input_to_candidate)

        new_state = update * state + (1 - update) * candidate
        return new_state, new_state
