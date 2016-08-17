import tensorflow as tf

def attention_decoder(decoder_inputs, initial_state, attention_objects,
                      embedding_size, cell, output_size=None,
                      loop_function=None, dtype=tf.float32, scope=None):

    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(scope or "attention_decoder"):
        batch_size = tf.shape(decoder_inputs[0])[0]    # Needed for reshaping.

        # do manualy broadcasting of the initial state if we want it
        # to be the same for all inputs
        if len(initial_state.get_shape()) == 1:
            state_size = initial_state.get_shape()[0].value
            initial_state = tf.reshape(tf.tile(initial_state,
                                               tf.shape(decoder_inputs[0])[:1]),
                                       [-1, state_size])

        state = initial_state
        outputs = []
        prev = None

        attns = [a.initialize(batch_size, dtype) for a in attention_objects]

        states = []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right
            # size.

            # inp je batch x embedding_size
            # attns jsou batch x cokoli leze z attention objektu

            #x = tf.nn.seq2seq.linear([inp] + attns, embedding_size, True)
            x = tf.concat(1, [inp] + attns)

            # Run the RNN.
            # When using GRU cells, these two are the same.
            cell_output, new_state = cell(x, state)
            state = new_state

            states.append(state)
            # Run the attention mechanism.
            attns = [a.attention(state) for a in attention_objects]

            if attns:
                with tf.variable_scope("AttnOutputProjection"):
                    output = tf.nn.seq2seq.linear([cell_output] + attns, output_size,
                                             True)
            else:
                output = cell_output

            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, states

class Attention(object):
    def __init__(self, attention_states, scope, dropout_placeholder,
                 input_weights=None, max_fertility=None):
        self.scope = scope
        self.attentions_in_time = []
        self.attention_states = tf.nn.dropout(attention_states,
                                              dropout_placeholder)
        self.input_weights = input_weights

        with tf.variable_scope(scope):
            self.attn_length = attention_states.get_shape()[1].value
            self.attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape
            # before.
            self.att_states_reshaped = tf.reshape(
                self.attention_states,
                [-1, self.attn_length, 1, self.attn_size])

            self.attention_vec_size = self.attn_size    # Size of query vectors
                                                        # for attention.

            k = tf.get_variable(
                "AttnW",
                [1, 1, self.attn_size, self.attention_vec_size])

            self.hidden_features = tf.nn.conv2d(self.att_states_reshaped, k,
                                                [1, 1, 1, 1], "SAME")

            self.v = tf.get_variable("AttnV", [self.attention_vec_size])

    def attention(self, query_state):
        """Put attention masks on att_states_reshaped
           using hidden_features and query.
        """

        with tf.variable_scope(self.scope+"/Attention"):
            y = tf.nn.seq2seq.linear(query_state, self.attention_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, self.attention_vec_size])

            s = self.get_logits(y)

            if self.input_weights is None:
                a = tf.nn.softmax(s)
            else:
                a_all = tf.nn.softmax(s) * self.input_weights
                norm = tf.reduce_sum(a_all, 1, keep_dims=True) + 1e-8
                a = a_all / norm

            self.attentions_in_time.append(a)

            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(tf.reshape(a, [-1, self.attn_length, 1, 1])
                              * self.att_states_reshaped, [1, 2])

            return tf.reshape(d, [-1, self.attn_size])

    def get_logits(self, y):
        # Attention mask is a softmax of v^T * tanh(...).
        return tf.reduce_sum(self.v * tf.tanh(self.hidden_features + y), [2, 3])

    def initialize(self, batch_size, dtype):
        batch_attn_size = tf.pack([batch_size, self.attn_size])
        initial = tf.zeros(batch_attn_size, dtype=dtype)
        # Ensure the second shape of attention vectors is set.
        initial.set_shape([None, self.attn_size])
        return initial


class CoverageAttention(Attention):
    def __init__(self, attention_states, scope, dropout_placeholder,
                 input_weights=None, max_fertility=5):

        super(CoverageAttention, self).__init__(attention_states, scope,
                                                dropout_placeholder,
                                                input_weights=input_weights,
                                                max_fertility=max_fertility)

        self.coverage_weights = tf.get_variable("coverage_matrix",
                                                [1, 1, 1, self.attn_size])
        self.fertility_weights = tf.get_variable("fertility_matrix",
                                                 [1, 1, self.attn_size])
        self.max_fertility = max_fertility

        self.fertility = 1e-8 + self.max_fertility * tf.sigmoid(
            tf.reduce_sum(self.fertility_weights * self.attention_states, [2]))


    def get_logits(self, y):
        coverage = sum(
            self.attentions_in_time) / self.fertility * self.input_weights

        logits = tf.reduce_sum(
            self.v * tf.tanh(
                self.hidden_features + y + self.coverage_weights * tf.reshape(
                    coverage, [-1, self.attn_length, 1, 1])),
            [2, 3])

        return logits
