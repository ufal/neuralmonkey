import tensorflow as tf

from neuralmonkey.checking import assert_shape
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.logging import log
from neuralmonkey.nn.bidirectional_rnn_layer import BidirectionalRNNLayer
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell
from neuralmonkey.nn.pervasive_dropout_wrapper import PervasiveDropoutWrapper
from neuralmonkey.checking import assert_type
from neuralmonkey.vocabulary import Vocabulary

# tests: lint, mypy

# pylint: disable=too-many-instance-attributes


class FactoredEncoder(Attentive):
    """Implementation of a generic encoder that processes an arbitrary
    number of input sequences.
    """

    def __init__(self, max_input_len, vocabularies, data_ids, embedding_sizes,
                 rnn_size, **kwargs):
        """Construct a new instance of the factored encoder.

        Args:
            max_input_len: Maximum input length (longer sequences are trimmed)

            vocabularies: List of vocabularies indexed
            data_ids: List of data series IDs
            embedding_sizes: List of embedding sizes for each data series

            rnn_size: The size of the hidden state

        Keyword arguments:
            use_noisy_activations: Boolean flag whether to use noisy activation
                                   functions in RNN cells.
                                   (see neuralmonkey.nn.noisy_gru_cell) [False]

            use_pervasive_dropout: Boolean flag whether to use pervasive
                                   dropout
                                   (see arxiv.org/abs/1606.02891) [False]

            attention_type: The attention to use. [None]
            attention_fertility: Fertility for CoverageAttention (if used). [3]

            name: The name for this encoder. [sentence_encoder]
            dropout_keep_prob: 1 - Dropout probability [1]
        """
        attention_type = kwargs.get("attention_type", None)
        attention_fertility = kwargs.get("attention_fertility", 3)
        super().__init__(
            attention_type, attention_fertility=attention_fertility)
        for vocabulary in vocabularies:
            assert_type(self, 'vocabulary', vocabulary, Vocabulary)

        self.vocabularies = vocabularies
        self.data_ids = data_ids
        self.embedding_sizes = embedding_sizes

        self.max_input_len = max_input_len
        self.rnn_size = rnn_size

        self.name = kwargs.get("name", "sentence_encoder")
        self.dropout_keep_prob = kwargs.get("dropout_keep_prob", 1)

        self.use_noisy_activations = kwargs.get("use_noisy_activations", False)
        self.use_pervasive_dropout = kwargs.get("use_pervasive_dropout", False)

        log("Building encoder graph, name: '{}'.".format(self.name))
        with tf.variable_scope(self.name):
            self._create_encoder_graph()

            # Attention mechanism

            log("Encoder graph constructed.")

    @property
    def _attention_mask(self):
        return self.__attention_mask

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    def _get_rnn_cell(self):
        """Return the RNN cell for the encoder"""
        # pylint: disable=redefined-variable-type
        if self.use_noisy_activations:
            cell = NoisyGRUCell(self.rnn_size, self.is_training)
        else:
            cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)

        if self.use_pervasive_dropout:
            # pylint: disable=no-member, undefined-variable
            # TODO fix this
            shape = tf.concat(0, [tf.shape(self.inputs[0]), [rnn_size]])
            # TODO shape needs recomputing

            dropout_mask = tf.floor(tf.random_uniform(shape, 0.0, 1.0)
                                    + self.dropout_placeholder)

            scale = tf.inv(self.dropout_placeholder)
            cell = PervasiveDropoutWrapper(cell, dropout_mask, scale)

        return cell

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
            for i in range(self.max_input_len + 2)]

        sentence_lengths = tf.to_int64(sum(self.padding_weights))

        self.factor_inputs = {}
        factors = []

        for data_id, vocabulary, embedding_size in zip(
                self.data_ids, self.vocabularies, self.embedding_sizes):
            # Create data placehoders. The tensors' length is max_input_len+2
            # because we add explicit start and end symbols.
            prefix = ""
            if len(self.data_ids) > 1:
                prefix = "{}_".format(data_id)

            names = ["{}input_{}".format(prefix, i)
                     for i in range(self.max_input_len + 2)]

            inputs = [tf.placeholder(tf.int32, shape=[None], name=n)
                      for n in names]

            # Create embeddings for this factor and embed the placeholders
            # NOTE the initialization
            embeddings = tf.get_variable(
                "word_embeddings", shape=[len(vocabulary), embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

            embedded_inputs = [tf.nn.embedding_lookup(embeddings, i)
                               for i in inputs]

            dropped_embedded_inputs = [
                tf.nn.dropout(i, self.dropout_placeholder)
                for i in embedded_inputs]

            # Resulting shape is batch x embedding_size
            assert_shape(dropped_embedded_inputs, [None, embedding_size])
            factors.append(dropped_embedded_inputs)

            # Add inputs and weights to self to be able to feed them
            self.factor_inputs[data_id] = inputs

        # Concatenate all embedded factors into one tensor
        # Resulting shape is batch x sum(embedding_size)

        # factors is a 2D list of embeddings of dims [factor-type, time-step]
        # by doing zip(*factors), we get a list of (factor-type) embedding
        # tuples indexed by the time step
        concatenated_factors = [tf.concat(1, related_factors)
                                for related_factors in zip(*factors)]
        assert_shape(concatenated_factors[0],
                     [None, sum(self.embedding_sizes)])
        forward_gru, backward_gru = self._get_birnn_cells()

        bidi_layer = BidirectionalRNNLayer(forward_gru, backward_gru,
                                           concatenated_factors,
                                           sentence_lengths)

        self.outputs_bidi = bidi_layer.outputs_bidi
        self.encoded = bidi_layer.encoded

        self.__attention_tensor = tf.concat(1, [tf.expand_dims(o, 1)
                                                for o in self.outputs_bidi])
        self.__attention_tensor = tf.nn.dropout(self.__attention_tensor,
                                                self.dropout_placeholder)
        self.__attention_mask = tf.concat(
            1, [tf.expand_dims(w, 1) for w in self.padding_weights])

    # pylint: disable=too-many-locals
    def feed_dict(self, dataset, train=False):
        factors = {data_id: dataset.get_series(data_id)
                   for data_id in self.data_ids}

        # this method should be responsible for checking if the factored
        # sentences are of the same length

        fd = {}
        # we asume that all factors have equal word counts
        # this is removed as res should only contain placeholders as keys
        # res[self.sentence_lengths] = np.array(
        #     [min(self.max_input_len, len(s)) +
        #      2 for s in factors[self.data_ids[0]]])
        factor_vectors_and_weights = {
            data_id: vocabulary.sentences_to_tensor(factors[data_id],
                                                    self.max_input_len,
                                                    train_mode=train)
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
                    raise Exception("Sentence lenghts are not the same for"
                                    "different factors.")

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
