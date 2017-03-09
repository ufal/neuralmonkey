from typing import List, Optional, Any
from typeguard import check_argument_types

import tensorflow as tf

from neuralmonkey.checking import assert_shape
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.logging import log
from neuralmonkey.vocabulary import Vocabulary


# pylint: disable=too-many-instance-attributes
class FactoredEncoder(ModelPart, Attentive):
    """Generic encoder processing an arbitrary number of input sequences."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 max_input_len: int,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 embedding_sizes: List[int],
                 rnn_size: int,
                 dropout_keep_prob: float=1.0,
                 attention_type: Optional[Any]=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        """Construct a new instance of the factored encoder.

        Args:
            max_input_len: Maximum input length (longer sequences are trimmed)
            vocabularies: List of vocabularies indexed
            data_ids: List of data series IDs
            embedding_sizes: List of embedding sizes for each data series
            name: The name for this encoder. [sentence_encoder]
            rnn_size: The size of the hidden state

        Keyword arguments:
            attention_type: The attention to use. [None]
            attention_fertility: Fertility for CoverageAttention (if used). [3]
            dropout_keep_prob: 1 - Dropout probability [1]
        """
        Attentive.__init__(self, attention_type)
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        assert check_argument_types()

        self.vocabularies = vocabularies
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes

        self.max_input_len = max_input_len
        self.rnn_size = rnn_size

        self.dropout_keep_prob = dropout_keep_prob

        log("Building encoder graph, name: '{}'.".format(self.name))
        with tf.variable_scope(self.name):
            self._create_encoder_graph()
            log("Encoder graph constructed.")
    # pylint: enable=too-many-arguments

    @property
    def _attention_mask(self):
        return self.__attention_mask

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    def _get_rnn_cell(self):
        """Return the RNN cell for the encoder"""
        return tf.contrib.rnn.GRUCell(self.rnn_size)

    def _get_birnn_cells(self):
        """Return forward and backward RNN cells for the encoder"""
        forward = self._get_rnn_cell()
        backward = self._get_rnn_cell()

        return forward, backward

    # pylint: disable=too-many-locals
    def _create_encoder_graph(self):
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.padding_weights = [
            tf.placeholder(tf.float32, shape=[None], name="input_{}".format(i))
            for i in range(self.max_input_len + 1)]

        sentence_lengths = tf.to_int64(sum(self.padding_weights))

        self.factor_inputs = {}
        factors = []

        assert len(self.data_ids) == len(self.vocabularies) == len(
            self.embedding_sizes)

        for data_id, vocabulary, embedding_size in zip(
                self.data_ids, self.vocabularies, self.embedding_sizes):
            # Create data placeholders. The tensors' length is max_input_len+1
            # because we add explicit start and end symbols.
            prefix = ""
            if len(self.data_ids) > 1:
                prefix = "{}_".format(data_id)

            names = ["{}input_{}".format(prefix, i)
                     for i in range(self.max_input_len + 1)]

            inputs = [tf.placeholder(tf.int32, shape=[None], name=n)
                      for n in names]

            # Create embeddings for this factor and embed the placeholders
            # NOTE the initialization
            embeddings = tf.get_variable(
                "embeddings_{}".format(data_id), shape=[len(vocabulary),
                                                        embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

            embedded_inputs = [tf.nn.embedding_lookup(embeddings, i)
                               for i in inputs]

            dropped_embedded_inputs = [
                tf.nn.dropout(i, self.dropout_placeholder)
                for i in embedded_inputs]

            # Resulting shape is batch x embedding_size
            factors.append(dropped_embedded_inputs)

            # Add inputs and weights to self to be able to feed them
            self.factor_inputs[data_id] = inputs

        # Concatenate all embedded factors into one tensor
        # Resulting shape is batch x sum(embedding_size)

        # factors is a 2D list of embeddings of dims [factor-type, time-step]
        # by doing zip(*factors), we get a list of (factor-type) embedding
        # tuples indexed by the time step
        concatenated_factors = [tf.concat(related_factors, 1)
                                for related_factors in zip(*factors)]
        assert_shape(concatenated_factors[0],
                     [None, sum(self.embedding_sizes)])
        forward_gru, backward_gru = self._get_birnn_cells()

        stacked_factors = tf.stack(concatenated_factors, 1)

        self.outputs_bidi, encoded_tup = tf.nn.bidirectional_dynamic_rnn(
            forward_gru, backward_gru, stacked_factors,
            sentence_lengths, dtype=tf.float32)

        self.encoded = tf.concat(encoded_tup, 1)

        self.__attention_tensor = tf.concat(self.outputs_bidi, 2)
        self.__attention_tensor = tf.nn.dropout(self.__attention_tensor,
                                                self.dropout_placeholder)
        self.__attention_mask = tf.concat(
            [tf.expand_dims(w, 1) for w in self.padding_weights], 1)

    # pylint: disable=too-many-locals
    def feed_dict(self, dataset, train=False):
        factors = {data_id: dataset.get_series(data_id)
                   for data_id in self.data_ids}

        factor_vectors_and_weights = {
            data_id: vocabulary.sentences_to_tensor(factors[data_id],
                                                    self.max_input_len,
                                                    train_mode=train,
                                                    add_start_symbol=True,
                                                    add_end_symbol=True)
            for data_id, vocabulary in zip(self.data_ids, self.vocabularies)}

        # check input lengths
        lengths = []
        paddings = None

        for _, (_, padding_weights) in factor_vectors_and_weights.items():
            paddings = padding_weights

            if len(lengths) == 0:
                lengths = [sum(p) for p in padding_weights]
            else:
                lengths_this = [sum(p) for p in padding_weights]

                if lengths_this != lengths:
                    raise Exception("Sentence lengths are not the same for"
                                    "different factors.")

        fd = {}
        for data_id in self.data_ids:
            inputs = self.factor_inputs[data_id]
            vectors, _ = factor_vectors_and_weights[data_id]
            for words_plc, words_tensor in zip(inputs, vectors):
                fd[words_plc] = words_tensor

        for plc, padding in zip(self.padding_weights, paddings):
            fd[plc] = padding

        if train:
            fd[self.dropout_placeholder] = self.dropout_keep_prob
        else:
            fd[self.dropout_placeholder] = 1.0
        fd[self.is_training] = train

        return fd
