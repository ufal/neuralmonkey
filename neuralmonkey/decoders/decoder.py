import math

import tensorflow as tf
import numpy as np

class Decoder(object):
"""A class that manages parts of the computation graph that are
used for the decoding.
"""

    def __init__(self, encoders, vocabulary, **kwargs):
        """Creates a new instance of the decoder

        Required arguments:
            encoders: List of encoders whose outputs will be decoded
            vocabulary: Output vocabulary

        Keyword arguments:
            embedding_size: Size of embedding vectors. Default 200
            max_output: Maximum length of the output. Default 20
            rnn_size: When projection is used, this is the size of the projected
                      vector

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

        self.max_output = kwargs.get("max_output", 20)
        self.embedding_size = kwargs.get("embedding_size", 200)
        self.name = kwargs.get("name", "decoder")

        self.use_attention = kwargs.get("use_attention", False)
        self.reuse_word_embeddings = kwargs.get("reuse_word_embeddings", False)
        self.project_encoder_outputs = kwargs.get("project_encoder_outputs",
                                                  False)

        if self.project_encoder_outputs:
            self.rnn_size = kwargs.get("rnn_size", 200)
        else:
            self.rnn_size = sum([e.encoded for e in self.encoders])


        ### Initialize model

        state = self.initial_state()

        self.w, self.b = self.state_to_output()
        self.embedding_matrix = self.input_embeddings()

        cell = self.get_rnn_cell()
        attention_objects = self.collect_attention_objects(self.encoders)

        self.train_inputs, train_targets, train_weights \
            = self.training_placeholders()


        ### Perform computation

        embedded_train_inputs = self.embed_inputs(self.train_inputs)

        self.train_rnn_outputs, _ = attention_decoder(
            embedded_train_inputs, state, attention_objects,
            self.embedding_size, cell)

        runtime_inputs = self.runtime_inputs()
        loop_function = self.get_loop_function()

        self.runtime_rnn_outputs, _ = attention_decoder(
            runtime_inputs, state, attention_objects, self.embedding_size,
            cell, loop_function=loop_function)


        ### KONEC decoder scope

        _, train_logits = self.decode(self.train_rnn_outputs)
        runtime_decoded, runtime_logits = self.decode(self.runtime_rnn_outputs)

        self.train_loss = tf.nn.seq2seq.sequence_loss(
            train_logits, train_targets, train_weights, self.vocabulary_size)

        self.runtime_loss = tf.nn.seq2seq.sequence_loss(
            runtime_logits, train_targets, train_weights, self.vocabulary_size)


        ### Summaries

        self.init_summaries()


    @property
    def vocabulary_size(self):
        return len(self.vocabulary)


    def init_summaries(self):
        """Initialize the summaries of the decoder

        TensorBoard summaries are collected into the following
        collections:

        - summary_train: collects statistics from the train-time
        - sumarry_val: collects OAstatistics while being tested on the
                 development data
        """
        tf.scalar_summary("train_loss_with_decoded_inputs", self.runtime_loss,
                          collections=["summary_train"])

        tf.scalar_summary("train_optimization_cost", self.train_loss,
                          collections=["summary_train"])


    def decode(self, rnn_states):
        """Decodes a sequence from a list of hidden states

        Arguments:
            rnn_outputs: hidden states
        """
        logits = []
        decoded = []

        for state in rnn_states:
            output_activation = self.logit_function(output)

            logits.append(output_activation)

            # we don"t want to generate padding
            decoded.append(tf.argmax(output_activation[:, 1:], 1) + 1)

        return decoded, logits



    def training_placeholders(self):
        """Defines data placeholders for training the decoder"""

        inputs = [tf.placeholder(tf.int64, [None], name="decoder_{}".format(i))
                  for i in range(self.max_output + 2)]

        targets = inputs[1:]

        # one less than inputs
        weights = [tf.placeholder(tf.float32, [None])
                   for _ in range(len(targets))]

        return inputs, targets, weights



    def runtime_inputs(self):
        """Defines data inputs for running trained decoder"""

        go_symbols = tf.fill(tf.shape(embedded_train_inputs[0], 1))
        go_embeds = tf.nn.embedding_lookup(self.embedding_matrix, go_symbols)

        inputs = [self.dropout(go_embeds)]
        inputs += [None for _ in range(self.max_output_len)]

        return inputs


    def state_to_output(self):
        """Create variables for projection of states to output vectors"""

        weights = tf.get_variable("state_to_word_W", [self.rnn_size,
                                                      self.vocabulary_size])

        biases = tf.Variable(tf.fill([self.vocabulary_size],
                                     - math.log(self.vocabulary_size)),
                             name="state_to_word_b")

        return weights, biases



    def input_embeddings(self):
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes
        them from the first encoder
        """
        if self.reuse_word_embeddings:
            return self.encoders[0].word_embeddings

        return tf.get_variable([self.vocabulary_size, self.embedding_size],
                               name="word_embeddings")


    def embed_inputs(self, inputs):
        """Embed inputs using the decoder"s word embedding matrix

        Arguments:
            inputs: List of input words to be embedded
        """
        embedded = [tf.nn.embedding_lookup(self.embedding_matrix, o)
                    for o in inputs]

        return self.dropout(embedded)


    def initial_state(self):
        """Create the initial state of the decoder."""
        encoders_out = tf.concat(1, [e.encoded for e in self.encoders])

        if self.project_encoder_outputs:
            encoders_out = self.encoder_projection(encoders_out)

        return self.dropout(encoders_out)


    def encoder_projection(self, encoded_states):
        """Creates a projection of concatenated encoder states

        Arguments:
            encoded_states: Tensor of concatenated states of input encoders
                            (batch x sum(states))
            rnn_size: Size of the projected vector
        """

        input_size = encoded_states.get_shape()[1].value
        output_size = self.rnn_size

        weights = tf.get_variable("encoder_projection_W", [input_size,
                                                           output_size])
        biases = tf.Variable(tf.zeros([output_size]),
                             name="encoder_projection_b"))

        dropped_input = self.dropout(encoded_states)
        return tf.matmul(dropped_input, weights) + biases


    def dropout(self, var):
        """Perform dropout on a variable

        Arguments:
            var: The variable to perform the dropout on
        """
        return tf.nn.dropout(var, self.dropout_placeholder)


    def logit_function(self, state):
        """Compute logits on the vocabulary given the state

        Arguments:
            state: the state of the decoder
        """
        return tf.matmul(self.dropout(state), self.w) + self.b


    def get_loop_funciton(self):
        """Constructs a loop function for the decoder"""

        def basic_loop(previous_state, _):
            """Basic loop function. Projects state to logits, take the
            argmax of the logits, embed the word and perform dropout on the
            embedding vector.

            Arguments:
                previous_state: The state of the decoder
                i: Unused argument, number of the time step
            """
            output_activation = self.logit_function(previous_state)
            previous_word = tf.argmax(output_activation, 1)
            input_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                     previous_word)

            return self.dropout(input_embedding)

        return basic_loop


    def get_rnn_cell(self):
        """Returns a RNNCell object for this decoder"""

        return tf.nn.rnn_cell.GRUCell(self.rnn_size)


    def collect_attention_objects(self, encoders):
        """Collect attention objects of the given encoders

        Arguments:
            encoders: Encoders from which to take attention objects
        """
        if self.use_attention:
            attention_objects = [e.attention_object for e in encoders
                                 if e.attention_object]
        else:
            return []
