# tests: lint, mypy

import tensorflow as tf

class BidirectionalRNNLayer(object):
    """Bidirectional RNN Layer class - forward and backward RNN layers in one.
    """

    def __init__(self, forward_cell, backward_cell, inputs,
                 sentence_lengths_placeholder):
        """Creates new BiRNN layer.

        Args:
          cell - the type of the cell (LSTMCell, GRUCell, NoisyGRUCell, ...)
          inputs - a list of inputs to the layer
          sentence_lengths_placeholder - lengths of the sequences in inputs

        """
        with tf.variable_scope('forward'):
            self._outputs, self._last_state = tf.nn.rnn(
                cell=forward_cell,
                inputs=inputs,
                dtype=tf.float32,
                sequence_length=sentence_lengths_placeholder)

        with tf.variable_scope('backward'):
            outputs_rev_rev, self._last_state_rev = tf.nn.rnn(
                cell=backward_cell,
                inputs=_reverse_seq(inputs, sentence_lengths_placeholder),
                dtype=tf.float32,
                sequence_length=sentence_lengths_placeholder)

            self._outputs_rev = _reverse_seq(outputs_rev_rev,
                                             sentence_lengths_placeholder)


    @property
    def outputs_bidi(self):
        """Outputs of the bidirectional layer"""

        ## outputs and outputs_rev, both lists in time of shape batch x rnn_size
        ## concatenations have shape batch x (2 * rnn_size)
        return [tf.concat(1, [o1, o2]) for o1, o2 in zip(self._outputs,
                                                         self._outputs_rev)]

    @property
    def encoded(self):
        """Last state of the bidirectional layer"""
        return tf.concat(1, [self._last_state, self._last_state_rev])


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.

    Arguments:
        input_seq: Sequence of seq_len tensors of dimension
                   (batch_size, depth)
        lengths:   A tensor of dimension batch_size, containing lengths for
                   each sequence in the batch. If "None" is specified,
                   simply reverses the list.
    Returns:
        time-reversed sequence
    """
    if lengths is None:
        return list(reversed(input_seq))

    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))

    # Join into (time, batch_size, depth)
    s_joined = tf.pack(input_seq)

    # Reverse along dimension 0
    s_reversed = tf.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = tf.unpack(s_reversed)
    return result
