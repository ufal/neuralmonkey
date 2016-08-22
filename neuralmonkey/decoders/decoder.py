#tests: lint

import math
import tensorflow as tf
import numpy as np

from neuralmonkey.decoding_function import attention_decoder
from neuralmonkey.logging import log

class Decoder(object):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-instance-attributes,too-many-locals
    # Big decoder cannot be simpler. Not sure if refactoring
    # it into smaller units would be helpful
    # Some locals may be turned to attributes

    def __init__(self, encoders, vocabulary, data_id, **kwargs):
        """Creates a new instance of the decoder

        Required arguments:
            encoders: List of encoders whose outputs will be decoded
            vocabulary: Output vocabulary
            data_id: Identifier of the data series fed to this decoder

        Keyword arguments:
            embedding_size: Size of embedding vectors. Default 200
            max_output: Maximum length of the output. Default 20
            rnn_size: When projection is used or when no encoder is supplied,
                      this is the size of the projected vector.
            dropout_keep_prob: Dropout keep probability. Default 1 (no dropout)

        Flags:
            use_attention: Indicates whether to use attention from encoders
            reuse_word_embeddings: Boolean flag specifying whether to reuse
                                   word embeddings. If True, word embeddings
                                   from the first encoder will be used
            project_encoder_outputs: Boolean flag whether to project output
                                     states of encoders

        """
        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id

        self.max_output = kwargs.get("max_output", 20)
        self.embedding_size = kwargs.get("embedding_size", 200)
        self.name = kwargs.get("name", "decoder")
        dropout_keep_prob = kwargs.get("dropout_keep_prob", 1.0)

        self.use_attention = kwargs.get("use_attention", False)
        self.reuse_word_embeddings = kwargs.get("reuse_word_embeddings", False)
        self.project_encoder_outputs = kwargs.get("project_encoder_outputs",
                                                  False)

        log("Initializing decoder, name: '{}'".format(self.name))

        if self.project_encoder_outputs or len(self.encoders) == 0:
            self.rnn_size = kwargs.get("rnn_size", 200)
        else:
            self.rnn_size = sum(e.encoded.get_shape()[1].value
                                for e in self.encoders)

        ### Initialize model

        self.dropout_placeholder = tf.placeholder_with_default(
            tf.constant(dropout_keep_prob, tf.float32),
            shape=[], name="decoder_dropout_placeholder")

        state = self._initial_state()

        self.weights, self.biases = self._state_to_output()
        self.embedding_matrix = self._input_embeddings()

        self.train_inputs, self.train_weights = self._training_placeholders()
        train_targets = self.train_inputs[1:]

        self.go_symbols = tf.placeholder(tf.int32, shape=[None],
                                         name="decoder_go_symbols")

        cell = self._get_rnn_cell()
        attention_objects = self._collect_attention_objects(self.encoders)

        ### Construct the computation part of the graph

        embedded_train_inputs = self._embed_inputs(self.train_inputs[:-1])

        self.train_rnn_outputs, _ = attention_decoder(
            embedded_train_inputs, state, attention_objects,
            self.embedding_size, cell)

        # runtime methods and objects are used when no ground truth is provided
        # (such as during testing)
        runtime_inputs = self._runtime_inputs(self.go_symbols)
        loop_function = self._get_loop_function()

        ### Use the same variables for runtime decoding!
        tf.get_variable_scope().reuse_variables()

        self.runtime_rnn_outputs, _ = attention_decoder(
            runtime_inputs, state, attention_objects, self.embedding_size,
            cell, loop_function=loop_function)

        _, train_logits = self._decode(self.train_rnn_outputs)
        self.decoded, runtime_logits = self._decode(self.runtime_rnn_outputs)

        self.train_loss = tf.nn.seq2seq.sequence_loss(
            train_logits, train_targets, self.train_weights,
            self.vocabulary_size)

        self.runtime_loss = tf.nn.seq2seq.sequence_loss(
            runtime_logits, train_targets, self.train_weights,
            self.vocabulary_size)

        ### Learning step
        ### TODO was here only because of scheduled sampling.
        ### needs to be refactored out
        self.learning_step = tf.Variable(0, name="learning_step",
                                         trainable=False)

        ### Summaries
        self._init_summaries()

        log("Decoder initialized.")


    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    @property
    def cost(self):
        return self.train_loss


    def _initial_state(self):
        """Create the initial state of the decoder."""
        if len(self.encoders) == 0:
            return tf.zeros([self.rnn_size])

        encoders_out = tf.concat(1, [e.encoded for e in self.encoders])

        if self.project_encoder_outputs:
            encoders_out = self._encoder_projection(encoders_out)

        return self._dropout(encoders_out)


    def _encoder_projection(self, encoded_states):
        """Creates a projection of concatenated encoder states

        Arguments:
            encoded_states: Tensor of concatenated states of input encoders
                            (batch x sum(states))
        """

        input_size = encoded_states.get_shape()[1].value
        output_size = self.rnn_size

        weights = tf.get_variable("encoder_projection_W", [input_size,
                                                           output_size])
        biases = tf.Variable(tf.zeros([output_size]),
                             name="encoder_projection_b")

        dropped_input = self._dropout(encoded_states)
        return tf.matmul(dropped_input, weights) + biases


    def _dropout(self, var):
        """Perform dropout on a variable

        Arguments:
            var: The variable to perform the dropout on
        """
        return tf.nn.dropout(var, self.dropout_placeholder)


    def _state_to_output(self):
        """Create variables for projection of states to output vectors"""

        weights = tf.Variable(
            tf.random_uniform([self.rnn_size, self.vocabulary_size], -0.5, 0.5),
            name="state_to_word_W")

        biases = tf.Variable(
            tf.fill([self.vocabulary_size], - math.log(self.vocabulary_size)),
            name="state_to_word_b")

        return weights, biases


    def _input_embeddings(self):
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes
        them from the first encoder
        """
        if self.reuse_word_embeddings:
            return self.encoders[0].word_embeddings

        return tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size],
                              -0.5, 0.5),
            name="word_embeddings")


    def _training_placeholders(self):
        """Defines data placeholders for training the decoder"""

        inputs = [tf.placeholder(tf.int64, [None], name="decoder_{}".format(i))
                  for i in range(self.max_output + 2)]

        # one less than inputs
        weights = [tf.placeholder(tf.float32, [None],
                                  name="decoder_padding_weights_{}".format(i))
                   for i in range(self.max_output + 1)]

        return inputs, weights


    def _get_rnn_cell(self):
        """Returns a RNNCell object for this decoder"""

        return tf.nn.rnn_cell.GRUCell(self.rnn_size)


    def _collect_attention_objects(self, encoders):
        """Collect attention objects of the given encoders

        Arguments:
            encoders: Encoders from which to take attention objects
        """
        if self.use_attention:
            return [e.attention_object for e in encoders if e.attention_object]
        else:
            return []


    def _embed_inputs(self, inputs):
        """Embed inputs using the decoder"s word embedding matrix

        Arguments:
            inputs: List of (batched) input words to be embedded
        """
        embedded = [tf.nn.embedding_lookup(self.embedding_matrix, o)
                    for o in inputs]
        return [self._dropout(e) for e in embedded]


    def _runtime_inputs(self, go_symbols):
        """Defines data inputs for running trained decoder

        Arguments:
            go_symbols: Tensor of go symbols. (Shape [batch])
        """
        go_embeds = tf.nn.embedding_lookup(self.embedding_matrix, go_symbols)

        inputs = [go_embeds]
        inputs += [None for _ in range(self.max_output)]

        return inputs


    def _get_loop_function(self):
        """Constructs a loop function for the decoder"""

        def basic_loop(previous_state, _):
            """Basic loop function. Projects state to logits, take the
            argmax of the logits, embed the word and perform dropout on the
            embedding vector.

            Arguments:
                previous_state: The state of the decoder
                i: Unused argument, number of the time step
            """
            output_activation = self._logit_function(previous_state)
            previous_word = tf.argmax(output_activation, 1)
            input_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                     previous_word)

            return self._dropout(input_embedding)

        return basic_loop


    def _logit_function(self, state):
        """Compute logits on the vocabulary given the state

        Arguments:
            state: the state of the decoder
        """
        return tf.matmul(self._dropout(state), self.weights) + self.biases


    def _decode(self, rnn_states):
        """Decodes a sequence from a list of hidden states

        Arguments:
            rnn_states: hidden states
        """
        logits = [self._logit_function(s) for s in rnn_states]
        decoded = [tf.argmax(l[:, 1:], 1) + 1 for l in logits]

        return decoded, logits


    def _init_summaries(self):
        """Initialize the summaries of the decoder

        TensorBoard summaries are collected into the following
        collections:

        - summary_train: collects statistics from the train-time
        """
        tf.scalar_summary("train_loss_with_decoded_inputs", self.runtime_loss,
                          collections=["summary_train"])

        tf.scalar_summary("train_optimization_cost", self.train_loss,
                          collections=["summary_train"])


    def feed_dict(self, dataset, train=False):
        """Populate the feed dictionary for the decoder object

        Decoder placeholders:
            decoder_{x} for x in range(max_output+2):
                Training data placeholders. Starts with <s> and ends with </s>

            decoder_padding_weights{x} for x in range(max_output+1):
                Weights used for padding. (Float) tensor of ones and zeros.
                This tensor is one-item shorter than the other one since the
                decoder does not produce the first <s>.

            dropout_placeholder: Scalar placeholder for dropout probability.
                Has value 'dropout_keep_prob' from the constructor or 1
                in case we are decoding at run-time
        """
        # pylint: disable=invalid-name
        # fd is the common name for feed dictionary
        fd = {}
        fd[self.go_symbols] = np.ones(len(dataset))

        sentences = dataset.get_series(self.data_id, allow_none=True)

        if sentences is not None:
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences, self.max_output)

            for placeholder, weight in zip(self.train_weights, weights):
                fd[placeholder] = weight

            for placeholder, tensor in zip(self.train_inputs, inputs):
                fd[placeholder] = tensor

        if not train:
            fd[self.dropout_placeholder] = 1.0

        return fd
