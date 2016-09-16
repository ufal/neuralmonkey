import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log
from neuralmonkey.nn.bidirectional_rnn_layer import BidirectionalRNNLayer
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.pervasive_dropout_wrapper import PervasiveDropoutWrapper
from neuralmonkey.checking import assert_type
from neuralmonkey.vocabulary import Vocabulary

# tests: mypy

class FactoredEncoder(object):
    def __init__(self, max_input_len, vocabularies, data_ids, embedding_sizes,
                 rnn_size, dropout_keep_p=0.5, use_noisy_activations=False,
                 use_pervasive_dropout=False, attention_type=None,
                 attention_fertility=3, name="sentence_encoder"):

        self.name = name
        self.max_input_len = max_input_len
        for vocabulary in vocabularies:
            assert_type(self, 'vocabulary', vocabulary, Vocabulary)
        self.vocabularies = vocabularies
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes
        self.rnn_size = rnn_size
        self.max_input_len = max_input_len
        self.dropout_keep_p = dropout_keep_p
        self.use_noisy_activations = use_noisy_activations
        self.use_pervasive_dropout = use_pervasive_dropout
        self.attention_type = attention_type
        self.attention_fertility = attention_fertility

        # TODO: checking.py expects a self.data_id field so we add one
        # for compatability but ideally checking.py should be refactored.
        self.data_id = self.data_ids[0]

        log("Initializing sentence encoder, name: \"{}\"".format(name))
        with tf.variable_scope(name):
            self.dropout_placeholder = tf.placeholder(tf.float32,
                                                      name="dropout")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.factor_inputs = {data_id: [tf.placeholder(tf.int32, shape=[None],
                                                           name="{}_input_{}".format(data_id, i))
                                            for i in range(max_input_len + 2)]
                                  for data_id in data_ids}

            self.factor_weight_ins =  {data_id: [tf.placeholder(tf.float32, shape=[None],
                                                                name="{}_input_{}".format(data_id, i))
                                                for i in range(max_input_len + 2)]
                                       for data_id in self.data_ids}

            self.weight_tensor = tf.concat(1, [tf.expand_dims(w, 1)
                                               for w in self.factor_weight_ins[self.data_ids[0]]])

            self.sentence_lengths = tf.to_int64(sum(self.factor_weight_ins[self.data_ids[0]]))

            self.word_embeddings = {data_id: tf.Variable(tf.random_uniform(
                                                [len(vocabulary), embedding_size], -1.0, 1.0))
                                    for data_id, vocabulary, embedding_size
                                        in zip(self.data_ids, self.vocabularies, self.embedding_sizes)}

            embedded_factors = {data_id: [tf.nn.embedding_lookup(self.word_embeddings[data_id], i)
                                          for i in factor]
                               for data_id, factor in self.factor_inputs.items()}
            dropped_embedded_factors = {data_id: [
                                            tf.nn.dropout(i, self.dropout_placeholder)
                                            for i in embedded_factor]
                                       for data_id, embedded_factor in embedded_factors.items()}
            concatenated_factors = [tf.concat(concat_dim=1, values=related_factors)
                                   for related_factors in zip(*[dropped_embedded_factors[data_id]
                                                                for data_id in self.data_ids])]

            if use_noisy_activations:
                self.forward_gru = NoisyGRUCell(rnn_size, self.is_training)
                self.backward_gru = NoisyGRUCell(rnn_size, self.is_training)
            else:
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
                                               concatenated_factors,
                                               self.sentence_lengths)

            self.outputs_bidi = bidi_layer.outputs_bidi
            self.encoded = bidi_layer.encoded

            self.attention_tensor = tf.concat(1, [tf.expand_dims(o, 1)
                                                  for o in self.outputs_bidi])

            self.attention_object = attention_type(
                self.attention_tensor, scope="attention_{}".format(name),
                dropout_placeholder=self.dropout_placeholder,
                input_weights=self.weight_tensor,
                max_fertility=attention_fertility) if attention_type else None

            log("Sentence encoder initialized")


    def feed_dict(self, dataset, train=False):
        factors = {data_id: dataset.get_series(data_id) for data_id in self.data_ids}

        res = {}
        # we asume that all factors have equal word counts
        # this is removed as res should only contain placeholders as keys
        # res[self.sentence_lengths] = np.array(
        #     [min(self.max_input_len, len(s)) + 2 for s in factors[self.data_ids[0]]])

        factor_vectors_and_weights = {data_id: vocabulary.sentences_to_tensor(
                                        factors[data_id], self.max_input_len, train=train)
                                      for data_id, vocabulary in zip(self.data_ids, self.vocabularies)}

        for data_id in self.data_ids:
            inputs = self.factor_inputs[data_id]
            vectors, _ = factor_vectors_and_weights[data_id]
            for words_plc, words_tensor in zip(inputs, vectors):
                res[words_plc] = words_tensor

        for data_id, weight_ins in self.factor_weight_ins.items():
            sentences = factors[data_id]
            res[weight_ins[0]] = np.ones(len(sentences)) 

        for data_id in self.data_ids:
            _, weights = factor_vectors_and_weights[data_id]
            weight_ins = self.factor_weight_ins[data_id]
            for weights_plc, weight_vector in zip(weight_ins[1:], weights):
                res[weights_plc] = weight_vector

        if train:
            res[self.dropout_placeholder] = self.dropout_keep_p
        else:
            res[self.dropout_placeholder] = 1.0
        res[self.is_training] = train

        return res
