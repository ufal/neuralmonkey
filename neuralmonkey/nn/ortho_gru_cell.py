import tensorflow as tf


# pylint: disable=too-few-public-methods
class OrthoGRUCell(tf.contrib.rnn.GRUCell):
    """Classic GRU cell but initialized using random orthogonal matrices."""

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 bias_initializer=None):
        tf.contrib.rnn.GRUCell.__init__(
            self, num_units, activation, reuse, tf.orthogonal_initializer(),
            bias_initializer)

    def __call__(self, inputs, state, scope="OrthoGRUCell"):
        return tf.contrib.rnn.GRUCell.__call__(self, inputs, state, scope)


# Note that tensorflow does not like when the typing annotations are present.
class NematusGRUCell(tf.contrib.rnn.GRUCell):

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
                kernel_initializer=tf.orthogonal_initializer(),
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
                name="state_proj")

            candidate = self._activation(
                state_to_candidate * reset + input_to_candidate)

        new_state = update * state + (1 - update) * candidate
        return new_state, new_state
