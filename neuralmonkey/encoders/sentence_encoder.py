import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.nn.bidirectional_rnn_layer import BidirectionalRNNLayer
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.pervasive_dropout_wrapper import PervasiveDropoutWrapper
from neuralmonkey.checking import assert_type
from neuralmonkey.vocabulary import Vocabulary

# tests: mypy

class SentenceEncoder(object):
    def __init__(self, max_input_len, vocabulary, data_id, embedding_size,
                 rnn_size, dropout_keep_p=0.5, use_noisy_activations=False,
                 use_pervasive_dropout=False, attention_type=None,
                 attention_fertility=3, name="sentence_encoder",
                 parent_encoder=None):

        self.name = name
        self.max_input_len = max_input_len
        assert_type(self, 'vocabulary', vocabulary, Vocabulary)
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.max_input_len = max_input_len
        self.dropout_keep_p = dropout_keep_p
        self.use_noisy_activations = use_noisy_activations
        self.use_pervasive_dropout = use_pervasive_dropout
        self.attention_type = attention_type
        self.attention_fertility = attention_fertility

        assert_type(self, 'parent_encoder', parent_encoder, SentenceEncoder,
                    can_be_none=True)
        self.parent_encoder = parent_encoder

        log("Initializing sentence encoder, name: \"{}\"".format(name))
        with tf.variable_scope(name):
            self.dropout_placeholder = tf.placeholder(tf.float32,
                                                      name="dropout")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.inputs = [tf.placeholder(tf.int32, shape=[None],
                                          name="input_{}".format(i))
                           for i in range(max_input_len + 2)]

            self.weight_ins = [tf.placeholder(tf.float32, shape=[None],
                                              name="input_{}".format(i))
                               for i in range(max_input_len + 2)]

            self.weight_tensor = tf.concat(1, [tf.expand_dims(w, 1)
                                               for w in self.weight_ins])

            self.sentence_lengths = tf.to_int64(sum(self.weight_ins))

            if parent_encoder:
                self.word_embeddings = parent_encoder.word_embeddings
            else:
                self.word_embeddings = tf.Variable(tf.random_uniform(
                    [len(vocabulary), embedding_size], -1.0, 1.0))

            embedded_inputs = [tf.nn.embedding_lookup(self.word_embeddings, i)
                               for i in self.inputs]
            dropped_embedded_inputs = [
                tf.nn.dropout(i, self.dropout_placeholder)
                for i in embedded_inputs]

            if parent_encoder:
                self.forward_gru = parent_encoder.forward_gru
                self.backward_gru = parent_encoder.backward_gru
            else:
                if use_noisy_activations:
                    self.forward_gru = NoisyGRUCell(rnn_size, self.is_training)
                    self.backward_gru = NoisyGRUCell(rnn_size, self.is_training)
                else:
                    ### this is used most of the time...
                    self.forward_gru = tf.nn.rnn_cell.GRUCell(rnn_size)
                    self.backward_gru = tf.nn.rnn_cell.GRUCell(rnn_size)

            if use_pervasive_dropout:

                # create dropout mask (shape batch x rnn_size)
                # floor (random uniform + dropout_keep)

                shape = tf.concat(0, [tf.shape(self.inputs[0]), [rnn_size]])

                forward_dropout_mask = tf.floor(
                    tf.random_uniform(shape, 0.0, 1.0) + self.dropout_placeholder)

                backward_dropout_mask = tf.floor(
                    tf.random_uniform(shape, 0.0, 1.0) + self.dropout_placeholder)

                scale = tf.inv(self.dropout_placeholder)

                self.forward_gru = PervasiveDropoutWrapper(
                    self.forward_gru, forward_dropout_mask, scale)
                self.backward_gru = PervasiveDropoutWrapper(
                    self.backward_gru, backward_dropout_mask, scale)




            bidi_layer = BidirectionalRNNLayer(self.forward_gru,
                                               self.backward_gru,
                                               dropped_embedded_inputs,
                                               self.sentence_lengths)

            self.outputs_bidi = bidi_layer.outputs_bidi
            self.encoded = bidi_layer.encoded

            # attention tensor is of shape batch x time x (2*rnn_size)
            self.attention_tensor = tf.concat(1, [tf.expand_dims(o, 1)
                                                  for o in self.outputs_bidi])

            self.attention_object = attention_type(
                self.attention_tensor, scope="attention_{}".format(name),
                dropout_placeholder=self.dropout_placeholder,
                input_weights=self.weight_tensor,
                max_fertility=attention_fertility) if attention_type else None

            log("Sentence encoder initialized")


    def feed_dict(self, dataset, train=False):
        sentences = dataset.get_series(self.data_id)

        res = {}
        res[self.sentence_lengths] = np.array(
            [min(self.max_input_len, len(s)) + 2 for s in sentences])

        vectors, weights = self.vocabulary.sentences_to_tensor(
            sentences, self.max_input_len, train=train)

        for words_plc, words_tensor in zip(self.inputs, vectors):
            res[words_plc] = words_tensor

        res[self.weight_ins[0]] = np.ones(len(sentences))

        for weights_plc, weight_vector in zip(self.weight_ins[1:], weights):
            res[weights_plc] = weight_vector

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_p
        else:
            res[self.dropout_placeholder] = 1.0
        res[self.is_training] = train

        return res
