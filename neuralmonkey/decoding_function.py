"""
Module which implements decoding functions using multiple attentions
for RNN decoders.

See http://arxiv.org/abs/1606.07481
"""
from abc import ABCMeta
from typing import Any, Dict, Tuple, NamedTuple

import tensorflow as tf
from neuralmonkey.nn.projection import linear
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell

# pylint: disable=invalid-name
AttentionLoopState = NamedTuple("AttentionLoopState",
                                [("contexts", tf.TensorArray),
                                 ("weights", tf.TensorArray)])
# pylint: enable=invalid-name


class BaseAttention(metaclass=ABCMeta):
    def __init__(self,
                 scope: str,
                 attention_states: tf.Tensor,
                 attention_state_size: int,
                 input_weights: tf.Tensor=None) -> None:
        self.scope = scope
        self.attention_states = attention_states
        self.attention_state_size = attention_state_size
        self.input_weights = input_weights

        self._histories = {}  # type: Dict[str, tf.Tensor]

    def attention(self,
                  decoder_state: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: Any, step: tf.Tensor) -> Tuple[tf.Tensor, Any]:
        """Get context vector for given decoder state."""
        raise NotImplementedError("Abstract method")

    def initial_loop_state(self) -> Any:
        """Get initial loop state for the attention object."""
        raise NotImplementedError("Abstract method")

    @property
    def histories(self) -> Dict[str, tf.Tensor]:
        return self._histories

    def finalize_loop(self, key: str, last_loop_state: Any) -> None:
        raise NotImplementedError("Abstract method")


class Attention(BaseAttention):
    # pylint: disable=unused-argument,too-many-instance-attributes
    # pylint: disable=too-many-arguments

    # For maintaining the same API as in CoverageAttention

    def __init__(self,
                 attention_states: tf.Tensor,
                 scope: str,
                 attention_state_size: int = None,
                 input_weights: tf.Tensor=None,
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
        super().__init__(
            scope, attention_states, attention_state_size, input_weights)

        self.attn_size = attention_states.get_shape()[2].value

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

    def initial_loop_state(self) -> AttentionLoopState:
        return AttentionLoopState(
            contexts=tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True,
                name="attention_contexts"),
            weights=tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True,
                name="attention_distributions"))

    def attention(self, decoder_state: tf.Tensor,
                  decoder_prev_state: tf.Tensor, _,
                  loop_state: AttentionLoopState, step: tf.Tensor) -> Tuple[
                      tf.Tensor, AttentionLoopState]:
        """put attention masks on att_states_reshaped
           using hidden_features and query.
        """

        with tf.variable_scope(self.scope + "/Attention") as varscope:
            # Sort-of a hack to get the matrix (bahdanau's W_a) in the linear
            # projection to be initialized this way. The biases are initialized
            # as zeros
            varscope.set_initializer(
                tf.random_normal_initializer(stddev=0.001))
            y = linear(decoder_state, self.attention_state_size,
                       scope=varscope)
            y = tf.reshape(y, [-1, 1, 1, self.attention_state_size])

            # pylint: disable=invalid-name
            # code copied from tensorflow. Suggestion: rename the variables
            # according to the Bahdanau paper
            s = self.get_logits(y, loop_state.weights)

            if self.input_weights is None:
                weights = tf.nn.softmax(s)
            else:
                weights_all = tf.nn.softmax(s) * self.input_weights
                norm = tf.reduce_sum(weights_all, 1, keep_dims=True) + 1e-8
                weights = weights_all / norm
            # pylint: enable=invalid-name

            # Now calculate the attention-weighted vector d.
            context = tf.reduce_sum(
                tf.expand_dims(tf.expand_dims(weights, -1), -1)
                * self.att_states_reshaped, [1, 2])
            context = tf.reshape(context, [-1, self.attn_size])

            next_contexts = loop_state.contexts.write(step, context)
            next_weights = loop_state.weights.write(step, weights)

            next_loop_state = AttentionLoopState(
                contexts=next_contexts,
                weights=next_weights)

            return context, next_loop_state

    def get_logits(self, y, _):
        # Attention mask is a softmax of v^T * tanh(...).
        return tf.reduce_sum(
            self.v * tf.tanh(self.hidden_features + y), [2, 3]) + self.v_bias

    def finalize_loop(self, key: str,
                      last_loop_state: AttentionLoopState) -> None:
        self.histories[key] = last_loop_state.weights


class CoverageAttention(Attention):

    # pylint: disable=too-many-arguments
    # Great objects require great number of parameters
    def __init__(self,
                 attention_states: tf.Tensor,
                 scope: str,
                 input_weights: tf.Tensor=None,
                 attention_fertility: int = 5) -> None:

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

    def get_logits(self, y, weights_in_time):

        weight_sum = tf.reduce_sum(weights_in_time.stack(), axis=0)

        coverage = weight_sum / self.fertility * self.input_weights

        logits = tf.reduce_sum(
            self.v * tf.tanh(
                self.hidden_features + y + self.coverage_weights *
                tf.expand_dims(tf.expand_dims(coverage, -1), -1)),
            [2, 3])

        return logits


# pylint: disable=too-few-public-methods
class RecurrentAttention(BaseAttention):
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
    # pylint: disable=unused-argument
    def __init__(self,
                 scope: str,
                 attention_states: tf.Tensor,
                 input_weights: tf.Tensor,
                 attention_state_size: int, **kwargs) -> None:
        super().__init__(
            scope, attention_states, attention_state_size, input_weights)

        self._state_size = attention_state_size
        self.attn_size = 2 * self._state_size

        self.fw_cell = OrthoGRUCell(self._state_size)
        self.bw_cell = OrthoGRUCell(self._state_size)
    # pylint: enable=unused-argument

    def initial_loop_state(self) -> None:
        return None

    # pylint: disable=unused-argument
    def attention(self,
                  decoder_state: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  _, loop_state: Any,
                  step: tf.Tensor) -> Tuple[tf.Tensor, Any]:

        with tf.variable_scope(self.scope + "/RecurrentAttn") as varscope:
            initial_state = linear(decoder_state, self._state_size, varscope)
            initial_state = tf.tanh(initial_state)

            # TODO dropout?
            # we'd need the train_mode and dropout_keep_prob parameters

            sentence_lengths = tf.to_int32(
                tf.reduce_sum(self.input_weights, 1))

            _, encoded_tup = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell, self.bw_cell, self.attention_states,
                sequence_length=sentence_lengths,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                dtype=tf.float32)

            return tf.concat(encoded_tup, 1), None

    def finalize_loop(self, key: str, last_loop_state: Any) -> None:
        pass
# pylint: disable=unused-argument,too-few-public-methods
