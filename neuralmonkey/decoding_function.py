"""
Module which implements decoding functions using multiple attentions
for RNN decoders.

See http://arxiv.org/abs/1606.07481
"""

import tensorflow as tf
from neuralmonkey.nn.projection import linear
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell


class Attention(object):
    # pylint: disable=unused-argument,too-many-instance-attributes
    # pylint: disable=too-many-arguments

    # For maintaining the same API as in CoverageAttention

    def __init__(self,
                 attention_states: tf.Tensor,
                 scope: str,
                 attention_state_size: int = None,
                 input_weights: tf.Tensor = None,
                 attention_fertility: int = None) -> None:
        """Create the attention object.

        Args:
            attention_states: A Tensor of shape (batch x time x state_size)
                with the output states of the encoder.
            scope: The name of the variable scope in the graph used by this
                attention object.
            attention_state_size: (Optional) the size of the attention inner
                state. If not supplied, the encoder rnn size will be used
                (x2 for bidirectional encoders)
            input_weights: (Optional) The padding weights on the input.
            attention_fertility: (Optional) For the Coverage attention
                compatibilty, maximum fertility of one word.
        """
        self.scope = scope
        self.logits_in_time = []  # type: List[tf.Tensor]
        self.attentions_in_time = []  # type: List[tf.Tensor]
        self.attention_states = attention_states
        self.input_weights = input_weights

        self.attn_size = attention_states.get_shape()[2].value
        self.attention_state_size = attention_state_size

        if self.attention_state_size is None:
            self.attention_state_size = self.attn_size

        with tf.variable_scope(scope):
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to
            # reshape before.
            self.att_states_reshaped = tf.expand_dims(self.attention_states, 2)

            # This variable corresponds to Bahdanau's U_a in the paper
            k = tf.get_variable(
                "AttnW", [1, 1, self.attn_size, self.attention_state_size],
                initializer=tf.random_normal_initializer(stddev=0.001))

            self.hidden_features = tf.nn.conv2d(self.att_states_reshaped, k,
                                                [1, 1, 1, 1], "SAME")

            # pylint: disable=invalid-name
            # see comments on disabling invalid names below
            self.v = tf.get_variable(
                name="AttnV",
                shape=[self.attention_state_size],
                initializer=tf.random_normal_initializer(stddev=.001))
            self.v_bias = tf.get_variable(
                "AttnV_b", [], initializer=tf.constant_initializer(0))

    def attention(self, query_state, prev_state, _) -> tf.Tensor:
        """Put attention masks on att_states_reshaped
           using hidden_features and query.
        """

        with tf.variable_scope(self.scope + "/Attention") as varscope:
            # Sort-of a hack to get the matrix (bahdanau's W_a) in the linear
            # projection to be initialized this way. The biases are initialized
            # as zeros
            varscope.set_initializer(
                tf.random_normal_initializer(stddev=0.001))
            y = linear(query_state, self.attention_state_size, scope=varscope)
            y = tf.reshape(y, [-1, 1, 1, self.attention_state_size])

            # pylint: disable=invalid-name
            # code copied from tensorflow. Suggestion: rename the variables
            # according to the Bahdanau paper
            s = self.get_logits(y)

            if self.input_weights is None:
                a = tf.nn.softmax(s)
            else:
                a_all = tf.nn.softmax(s) * self.input_weights
                norm = tf.reduce_sum(a_all, 1, keep_dims=True) + 1e-8
                a = a_all / norm

            self.logits_in_time.append(s)
            self.attentions_in_time.append(a)

            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(tf.expand_dims(tf.expand_dims(a, -1), -1)
                              * self.att_states_reshaped, [1, 2])

            return tf.reshape(d, [-1, self.attn_size])

    def get_logits(self, y):
        # Attention mask is a softmax of v^T * tanh(...).
        return tf.reduce_sum(
            self.v * tf.tanh(self.hidden_features + y), [2, 3]) + self.v_bias


class CoverageAttention(Attention):

    # pylint: disable=too-many-arguments
    # Great objects require great number of parameters
    def __init__(self, attention_states, scope,
                 input_weights=None, attention_fertility=5):

        super(CoverageAttention, self).__init__(
            attention_states, scope,
            input_weights=input_weights,
            attention_fertility=attention_fertility)

        self.coverage_weights = tf.get_variable("coverage_matrix",
                                                [1, 1, 1, self.attn_size])
        self.fertility_weights = tf.get_variable("fertility_matrix",
                                                 [1, 1, self.attn_size])
        self.attention_fertility = attention_fertility

        self.fertility = 1e-8 + self.attention_fertility * tf.sigmoid(
            tf.reduce_sum(self.fertility_weights * self.attention_states, [2]))

    def get_logits(self, y):
        coverage = sum(
            self.attentions_in_time) / self.fertility * self.input_weights

        logits = tf.reduce_sum(
            self.v * tf.tanh(
                self.hidden_features + y + self.coverage_weights *
                tf.expand_dims(tf.expand_dims(coverage, -1), -1)),
            [2, 3])

        return logits


class RecurrentAttention(object):
    """From article `Recurrent Neural Machine Translation
    `<https://arxiv.org/pdf/1607.08725v1.pdf>`_

    In time i of the decoder with state s_i-1, and encoder states h_j, we run
    a bidirectional RNN with initial state set to

    c_0 = tanh(V*s_i-1 + b_0)

    Then we run the GRU net (in paper just forward, we do bidi)
    and we get N+1 hidden states c_0 ... c_N

    to compute the context vector, they try either last state or mean of
    all the states. Last state was better in their experiments so that's what
    we're gonna use.
    """
    def __init__(self, attention_tensor, scope, input_weights, **kwargs):
        self._attention_tensor = attention_tensor
        self._scope = scope
        self._input_mask = input_weights

        if "attention_state_size" not in kwargs:
            raise ValueError("RecurrentAttention need attention_state_size"
                             " in kwargs")

        self._state_size = kwargs["attention_state_size"]
        self.attn_size = 2 * self._state_size

        self.fw_cell = OrthoGRUCell(self._state_size)
        self.bw_cell = OrthoGRUCell(self._state_size)

    def attention(self, query_state, prev_state, _):

        with tf.variable_scope(
                self._scope + "/RecurrentAttention") as varscope:
            initial_state = linear(query_state, self._state_size, varscope)
            initial_state = tf.tanh(initial_state)

            # TODO dropout?
            # we'd need the train_mode and dropout_keep_prob parameters

            sentence_lengths = tf.to_int32(tf.reduce_sum(self._input_mask, 1))

            _, encoded_tup = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell, self.bw_cell, self._attention_tensor,
                sequence_length=sentence_lengths,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                dtype=tf.float32)

            return tf.concat(1, encoded_tup)
