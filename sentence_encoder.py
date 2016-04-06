import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.models.rnn import rnn_cell, rnn, seq2seq

from noisy_gru_cell import NoisyGRUCell

class SentenceEncoder(object):
    def __init__(self, max_input_len, vocabulary, embedding_size, rnn_size, dropout_placeholder,
                 is_training, use_noisy_activations=False, name="sentence_encoder"):
        with tf.variable_scope(name):
            self.inputs = \
                    [tf.placeholder(tf.int32, shape=[None], name="input_{}".format(i)) for i in range(max_input_len + 2)]
            self.sentence_lengths = tf.placeholder(tf.int64, shape=[None], name="sequence_lengths")

            self.word_embeddings = tf.Variable(tf.random_uniform([len(vocabulary), embedding_size], -1.0, 1.0))

            embedded_inputs = [tf.nn.embedding_lookup(self.word_embeddings, input_) for input_ in self.inputs]

            dropped_embedded_inputs = [tf.nn.dropout(i, dropout_placeholder) for i in embedded_inputs]

            with tf.variable_scope('forward'):
                if use_noisy_activations:
                    gru = rnn_cell.GRUCell(rnn_size, input_size=embedding_size)
                else:
                    gru = NoisyGRUCell(rnn_size, is_training, input_size=embedding_size)
                outputs, last_state = rnn.rnn(
                    cell=gru,
                    inputs=dropped_embedded_inputs, dtype=tf.float32,
                    sequence_length=self.sentence_lengths)

            with tf.variable_scope('backward'):
                if use_noisy_activations:
                    gru_rev = rnn_cell.GRUCell(rnn_size, input_size=embedding_size)
                else:
                    gru_rev = NoisyGRUCell(rnn_size, is_training, input_size=embedding_size)
                outputs_rev_rev, last_state_rev = rnn.rnn(
                    cell=gru_rev,
                    inputs=self._reverse_seq(dropped_embedded_inputs, self.sentence_lengths), dtype=tf.float32,
                    sequence_length=self.sentence_lengths)

                outputs_rev=self._reverse_seq(outputs_rev_rev, self.sentence_lengths)

            outputs_bidi = [tf.concat(1, [o1, o2]) for o1, o2 in zip(outputs, reversed(outputs_rev))]

            self.encoded = tf.concat(1, [last_state, last_state_rev])

            self.attention_tensor = \
                    tf.concat(1, [tf.expand_dims(o, 1) for o in outputs_bidi])
                    #tf.transpose(tf.concat(1, [tf.expand_dims(o, 1) for o in outputs_bidi]), [0, 2, 1])

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
