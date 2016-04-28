import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, loop_function=None,
                      dtype=tf.float32, scope=None):

    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(scope or "attention_decoder"):
        batch_size = tf.shape(decoder_inputs[0])[0]    # Needed for reshaping.

        att_objects = [Attention(states, scope="att_{}".format(i)) for i, states in enumerate(attention_states)]

        state = initial_state
        outputs = []
        prev = None

        attns = [a.initialize(batch_size, dtype) for a in att_objects]

        states = []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            x = rnn_cell.linear([inp] + attns, cell.input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            states.append(state)
            # Run the attention mechanism.
            attns = [a.attention(state) for a in att_objects]

            if attns:
                with tf.variable_scope("AttnOutputProjection"):
                    output = rnn_cell.linear([cell_output] + attns, output_size, True)
            else:
                output = cell_output

            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, states

class Attention(object):
    def __init__(self, attention_states, scope):
        self.scope = scope
        with tf.variable_scope(scope):
            self.attn_length = attention_states.get_shape()[1].value
            self.attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            self.att_states_reshaped = tf.reshape(
                    attention_states, [-1, self.attn_length, 1, self.attn_size])
            self.attention_vec_size = self.attn_size    # Size of query vectors for attention.
            k = tf.get_variable("AttnW", [1, 1, self.attn_size, self.attention_vec_size])
            self.hidden_features = tf.nn.conv2d(self.att_states_reshaped, k, [1, 1, 1, 1], "SAME")
            self.v = tf.get_variable("AttnV", [self.attention_vec_size])

    def attention(self, query_state):
        """Put attention masks on att_states_reshaped using hidden_features and query."""
        with tf.variable_scope(self.scope+"/Attention"):
            y = rnn_cell.linear(query_state, self.attention_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, self.attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(
                    self.v * tf.tanh(self.hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                    tf.reshape(a, [-1, self.attn_length, 1, 1]) * self.att_states_reshaped,
                    [1, 2])
            return tf.reshape(d, [-1, self.attn_size])

    def initialize(self, batch_size, dtype):
        batch_attn_size = tf.pack([batch_size, self.attn_size])
        initial = tf.zeros(batch_attn_size, dtype=dtype)
        # Ensure the second shape of attention vectors is set.
        initial.set_shape([None, self.attn_size])
        return initial
