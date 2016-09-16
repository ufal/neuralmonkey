import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.nn.bidirectional_rnn_layer import BidirectionalRNNLayer
from neuralmonkey.checking import assert_type
from neuralmonkey.decoding_function import Attention
from neuralmonkey.vocabulary import Vocabulary

# tests: mypy

class VanillaSentenceEncoder(object):
    """Like the SentenceEncoder, but ignores the start/end tokens."""
    def __init__(self, max_input_len, vocabulary, data_id, embedding_size,
                 rnn_size,
                 dropout_keep_p=0.5,
                 name="sentence_encoder",
                 **kwargs):

        self.name = name
        self.max_input_len = max_input_len
        assert_type(self, 'vocabulary', vocabulary, Vocabulary)
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.max_input_len = max_input_len
        self.dropout_keep_p = dropout_keep_p

        log("Initializing sentence encoder, name: \"{}\"".format(name))
        with tf.variable_scope(name):
            self.dropout_placeholder = tf.placeholder(tf.float32,
                                                      name="dropout")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.inputs = [tf.placeholder(tf.int32, shape=[None],
                                          name="input_{}".format(i))
                           for i in range(max_input_len)]

            self.weight_ins = [tf.placeholder(tf.float32, shape=[None],
                                              name="input_{}".format(i))
                               for i in range(max_input_len)]

            self.weight_tensor = tf.concat(1, [tf.expand_dims(w, 1)
                                               for w in self.weight_ins])

            self.sentence_lengths = tf.to_int64(sum(self.weight_ins))

            self.word_embeddings = tf.Variable(tf.random_uniform(
                [len(vocabulary), embedding_size], -1.0, 1.0))

            embedded_inputs = [tf.nn.embedding_lookup(self.word_embeddings, i)
                               for i in self.inputs]
            dropped_embedded_inputs = [
                tf.nn.dropout(i, self.dropout_placeholder)
                for i in embedded_inputs]

            self.forward_gru = tf.nn.rnn_cell.GRUCell(rnn_size)
            self.backward_gru = tf.nn.rnn_cell.GRUCell(rnn_size)


            bidi_layer = BidirectionalRNNLayer(self.forward_gru,
                                               self.backward_gru,
                                               dropped_embedded_inputs,
                                               self.sentence_lengths)

            self.outputs_bidi = bidi_layer.outputs_bidi
            self.encoded = bidi_layer.encoded

            self.attention_tensor = tf.concat(1, [tf.expand_dims(o, 1)
                                                  for o in self.outputs_bidi])

            self.attention_object = Attention(
                self.attention_tensor, scope="attention_{}".format(name),
                dropout_placeholder=self.dropout_placeholder,
                input_weights=self.weight_tensor)

            log("Sentence encoder initialized")


    def feed_dict(self, dataset, train=False):
        sentences = dataset.get_series(self.data_id)

        fd = {}
        fd[self.sentence_lengths] = np.array(
            [min(self.max_input_len, len(s)) for s in sentences])

        vectors, weights = self.vocabulary.sentences_to_bare_tensor(
            sentences, self.max_input_len, train=train)

        for words_plc, words_tensor in zip(self.inputs, vectors):
            fd[words_plc] = words_tensor

        # fd[self.weight_ins[0]] = np.ones(len(sentences))

        for weights_plc, weight_vector in zip(self.weight_ins, weights):
            fd[weights_plc] = weight_vector

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_p
        else:
            fd[self.dropout_placeholder] = 1.0
        fd[self.is_training] = train

        return fd
