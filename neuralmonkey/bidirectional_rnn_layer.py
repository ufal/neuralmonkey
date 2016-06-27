import tensorflow as tf
from tensorflow.python.ops import array_ops

class BidirectionalRNNLayer(object):
    """
    Bidirectional RNN Layer class - forward and backward RNN layers in one.

    """


    def __init__(self, forward_cell, backward_cell, inputs, sentence_lengths_placeholder):
        """
        Creates new BiRNN layer.

        Args:
        cell - the type of the cell (LSTMCell, GRUCell, NoisyGRUCell, ...)
        inputs - a list of inputs to the layer
        sentence_lengths_placeholder - lengths of the sequences in inputs

        """


        with tf.variable_scope('forward'):
            outputs, last_state = tf.nn.rnn(
                cell=forward_cell,
                inputs=inputs,
                dtype=tf.float32,
                sequence_length=sentence_lengths_placeholder)


        with tf.variable_scope('backward'):
            outputs_rev_rev, last_state_rev = tf.nn.rnn(
                cell=backward_cell,
                inputs=self._reverse_seq(inputs, sentence_lengths_placeholder),
                dtype=tf.float32,
                sequence_length=sentence_lengths_placeholder)

            outputs_rev=self._reverse_seq(outputs_rev_rev, sentence_lengths_placeholder)

        self.outputs_bidi = [tf.concat(1, [o1, o2]) for o1, o2 in zip(outputs, outputs_rev)]
        self.encoded = tf.concat(1, [last_state, last_state_rev])


    def _reverse_seq(self, input_seq, lengths):
        """Reverse a list of Tensors up to specified lengths.

        Args:
            input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
            lengths:   A tensor of dimension batch_size, containing lengths for each
                       sequence in the batch. If "None" is specified, simply reverses
                       the list.

        Returns:
            time-reversed sequence
        """
        if lengths is None:
            return list(reversed(input_seq))

        for input_ in input_seq:
            input_.set_shape(input_.get_shape().with_rank(2))

        # Join into (time, batch_size, depth)
        s_joined = array_ops.pack(input_seq)

        # Reverse along dimension 0
        s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
        # Split again into list
        result = array_ops.unpack(s_reversed)
        return result
