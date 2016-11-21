import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.pervasive_dropout_wrapper import PervasiveDropoutWrapper
from neuralmonkey.checking import assert_type
from neuralmonkey.vocabulary import Vocabulary

# tests: mypy

class SentenceEncoder(object):
    def __init__(self, max_input_len, vocabulary, data_id, embedding_size,
                 rnn_size, name, dropout_keep_p=0.5,
                 use_noisy_activations=False, use_pervasive_dropout=False,
                 attention_type=None, attention_fertility=3,
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

            self.inputs = tf.placeholder(tf.int32, shape=[None, max_input_len + 2], name="encoder_inputs")
            self.sentence_lengths = tf.placeholder(tf.int64, shape=[None], name="encoder_input_lengths")

            # TODO remove this placeholder and refactor attention to work
            # with sentence_lengths
            self.weight_tensor = tf.placeholder(tf.float32, name="encoder_padding_weights")

            if parent_encoder:
                self.word_embeddings = parent_encoder.word_embeddings
            else:
                self.word_embeddings = tf.Variable(tf.random_uniform(
                    [len(vocabulary), embedding_size], -1.0, 1.0))

            embedded_inputs = tf.nn.embedding_lookup(self.word_embeddings, self.inputs)
            dropped_embedded_inputs = tf.nn.dropout(embedded_inputs, self.dropout_placeholder)

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


            outputs_bidi_t, encoded_t = tf.nn.bidirectional_dynamic_rnn(
                self.forward_gru, self.backward_gru,
                dropped_embedded_inputs, self.sentence_lengths,
                dtype=tf.float32)

            self.attention_tensor = tf.concat(2, outputs_bidi_t)
            self.encoded = tf.concat(1, encoded_t)

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

        # as sentences_to_tensor returns list of shape (max-len+2) x batch,
        # we need to transpose
        res[self.inputs] = list(zip(*vectors))

        begin_weights = np.ones(len(sentences))

        # shape of weights: (max-len+1) x batch
        # shape of weight tensor should be (max-len+2) x batch
        res[self.weight_tensor] = np.vstack((begin_weights, weights)).T

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_p
        else:
            res[self.dropout_placeholder] = 1.0
        res[self.is_training] = train

        return res
