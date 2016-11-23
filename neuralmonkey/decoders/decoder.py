#tests: lint

import tensorflow as tf
import numpy as np

from neuralmonkey.vocabulary import START_TOKEN
from neuralmonkey.logging import log

class Decoder(object):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
    # Big decoder cannot be simpler. Not sure if refactoring
    # it into smaller units would be helpful
    # Some locals may be turned to attributes

    def __init__(self, encoders, vocabulary, data_id, name, **kwargs):
        """Creates a new instance of the decoder

        Arguments:
            encoders: List of encoders whose outputs will be decoded
            vocabulary: Output vocabulary
            data_id: Identifier of the data series fed to this decoder

        Keyword arguments:
            embedding_size: Size of embedding vectors. Default 200
            max_output_len: Maximum length of the output. Default 20
            rnn_size: When projection is used or when no encoder is supplied,
                this is the size of the projected vector.
            dropout_keep_prob: Dropout keep probability. Default 1 (no dropout)
            use_attention: Boolean flag that indicates whether to use attention
                from encoders
            reuse_word_embeddings: Boolean flag specifying whether to
                reuse word embeddings. If True, word embeddings
                from the first encoder will be used
            project_encoder_outputs: Boolean flag whether to project output
                states of encoders
        """
        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.name = name

        self.max_output = kwargs.get("max_output_len", 20)
        self.embedding_size = kwargs.get("embedding_size", 200)
        dropout_keep_prob = kwargs.get("dropout_keep_prob", 1.0)

        self.use_attention = kwargs.get("use_attention", False)
        self.reuse_word_embeddings = kwargs.get("reuse_word_embeddings", False)

        if self.reuse_word_embeddings:
            self.embedding_size = self.encoders[0].embedding_size

            if "embedding_size" in kwargs:
                log("Warning: Overriding embedding_size parameter with reused"
                    " embeddings from the encoder.", color="red")

        self.project_encoder_outputs = kwargs.get("project_encoder_outputs",
                                                  False)

        log("Initializing decoder, name: '{}'".format(self.name))

        ### Learning step
        ### TODO was here only because of scheduled sampling.
        ### needs to be refactored out
        self.learning_step = tf.get_variable(
            "learning_step", [], initializer=tf.constant_initializer(0),
            trainable=False)

        if self.project_encoder_outputs or len(self.encoders) == 0:
            self.rnn_size = kwargs.get("rnn_size", 200)
        else:
            if "rnn_size" in kwargs:
                log("Warning: rnn_size attribute will not be used "
                    "without encoder projection!", color="red")

            self.rnn_size = sum(e.encoded.get_shape()[1].value
                                for e in self.encoders)

        ### Initialize model

        self.dropout_placeholder = tf.placeholder_with_default(
            tf.constant(dropout_keep_prob, tf.float32),
            shape=[], name="decoder_dropout_placeholder")

        state = self._initial_state()

        self.weights, self.biases = self._rnn_output_proj_params()

        self.embedding_matrix = self._input_embeddings()

        self.train_inputs, self.train_weights = self._training_placeholders()
        train_targets = self.train_inputs[1:]

        self.go_symbols = tf.placeholder(tf.int32, shape=[None],
                                         name="decoder_go_symbols")

        ### Construct the computation part of the graph

        embedded_train_inputs = self._embed_inputs(self.train_inputs[:-1])

        self.train_rnn_outputs, _ = self._attention_decoder(
            embedded_train_inputs, state)

        # runtime methods and objects are used when no ground truth is provided
        # (such as during testing)
        runtime_inputs = self._runtime_inputs(self.go_symbols)

        ### Use the same variables for runtime decoding!
        tf.get_variable_scope().reuse_variables()

        self.runtime_rnn_outputs, self.runtime_rnn_states = \
            self._attention_decoder(
                runtime_inputs, state, runtime_mode=True,
                summary_collections=["summary_val_plots"])

        val_plots_collection = tf.get_collection("summary_val_plots")
        self.summary_val_plots = (
            tf.merge_summary(val_plots_collection)
            if val_plots_collection else None
        )

        _, train_logits = self._decode(self.train_rnn_outputs)
        self.decoded, runtime_logits = self._decode(self.runtime_rnn_outputs)
        self.train_logprobs = [tf.nn.log_softmax(l) for l in train_logits]

        self.train_loss = tf.nn.seq2seq.sequence_loss(
            train_logits, train_targets, self.train_weights,
            self.vocabulary_size)

        self.runtime_loss = tf.nn.seq2seq.sequence_loss(
            runtime_logits, train_targets, self.train_weights,
            self.vocabulary_size)

        self.cross_entropies = tf.nn.seq2seq.sequence_loss_by_example(
            train_logits, train_targets, self.train_weights,
            self.vocabulary_size)

        # TODO [refactor] put runtime logits to self from the beginning
        self.runtime_logits = runtime_logits
        self.runtime_logprobs = [tf.nn.log_softmax(l) for l in runtime_logits]

        ### Summaries
        self._init_summaries()

        log("Decoder initialized.")


    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    @property
    def cost(self):
        return self.train_loss


    def top_k_runtime_logprobs(self, k_best):
        """Return the top runtime log probabilities calculated from runtime
        logits.

        Arguments:
            k_best: How many output items to return
        """
        ## the array is of tuples ([values], [indices])
        return [tf.nn.top_k(p, k_best) for p in self.runtime_logprobs]


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
        and applies a tanh activation

        Arguments:
            encoded_states: Tensor of concatenated states of input encoders
                            (batch x sum(states))
        """
        input_size = encoded_states.get_shape()[1].value
        output_size = self.rnn_size

        weights = tf.get_variable(
            "encoder_projection_W", [input_size, output_size],
            initializer=tf.random_normal_initializer(stddev=0.01))

        biases = tf.get_variable(
            "encoder_projection_b",
            initializer=tf.zeros_initializer([output_size]))

        dropped_input = self._dropout(encoded_states)
        return tf.tanh(tf.matmul(dropped_input, weights) + biases)


    def _dropout(self, var):
        """Perform dropout on a variable

        Arguments:
            var: The variable to perform the dropout on
        """
        return tf.nn.dropout(var, self.dropout_placeholder)


    def _input_embeddings(self):
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes
        them from the first encoder
        """
        if self.reuse_word_embeddings:
            return self.encoders[0].word_embeddings

        # NOTE In the Bahdanau paper, they say they initialized some weights
        # as orthogonal matrices, some by sampling from gauss distro with
        # stddev=0.001 and all other weight matrices as gaussian with
        # stddev=0.01. Embeddings were not among any of the special cases so
        # I assume that they initialized them as any other weight matrix.
        return tf.get_variable(
            "word_embeddings", [self.vocabulary_size, self.embedding_size],
            initializer=tf.random_normal_initializer(stddev=0.01))


    def _training_placeholders(self):
        """Defines data placeholders for training the decoder"""

        inputs = [tf.placeholder(tf.int64, [None], name="decoder_{}".format(i))
                  for i in range(self.max_output + 2)]

        # one less than inputs
        weights = [tf.placeholder(tf.float32, [None],
                                  name="decoder_padding_weights_{}".format(i))
                   for i in range(self.max_output + 1)]

        return inputs, weights


    def _rnn_output_proj_params(self):
        """Create parameters for projection of RNN outputs to vocabulary
        indices.

        This method should provide parameter shapes compatible with whatever
        the _get_rnn_output method is returning.
        """
        ctx_sizes = [a.attn_size for a in self._collect_attention_objects()]
        state_size = self.rnn_size + self.embedding_size + sum(ctx_sizes)

        weights = tf.get_variable(
            "state_to_word_W", [state_size, self.vocabulary_size],
            initializer=tf.random_normal_initializer(stddev=0.01)

        biases = tf.get_variable(
            "state_to_word_b",
            initializer=tf.zeros_initializer([self.vocabulary_size]))

        return weights, biases


    def _get_rnn_cell(self):
        """Returns a RNNCell object for this decoder"""
        return tf.nn.rnn_cell.GRUCell(self.rnn_size)


    def _collect_attention_objects(self):
        """Collect attention objects from encoders."""
        if not self.use_attention:
            return []
        return [e.attention_object for e in self.encoders if e.attention_object]


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


    def _loop_function(self, rnn_output):
        """Basic loop function. Projects state to logits, take the
        argmax of the logits, embed the word and perform dropout on the
        embedding vector.

        Arguments:
            rnn_output: The output of the decoder RNN
        """
        output_activation = self._logit_function(rnn_output)
        previous_word = tf.argmax(output_activation, 1)
        input_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                 previous_word)
        return self._dropout(input_embedding)


    def _logit_function(self, rnn_output):
        """Compute logits on the vocabulary given the state

        Arguments:
            rnn_output: the output of the decoder RNN
        """
        return tf.matmul(self._dropout(rnn_output), self.weights) + self.biases

    #pylint: disable=too-many-arguments
    # TODO reduce the number of arguments
    def _attention_decoder(self, inputs, initial_state, runtime_mode=False,
                           summary_collections=None, scope="attention_decoder"):
        """Run the decoder RNN.

        Arguments:
            inputs: The decoder inputs. If runtime_mode=True, only the first
                    input is used.
            initial_state: The initial state of the decoder.
            runtime_mode: Boolean flag whether the decoder is running in
                          runtime mode (with loop function).
            summary_collections: The list of summary collections to which
                                 the alignments are logged.
            scope: The variable scope to use with this function.
        """
        cell = self._get_rnn_cell()
        att_objects = self._collect_attention_objects()

        ## Broadcast the initial state to the whole batch if needed
        if len(initial_state.get_shape()) == 1:
            assert initial_state.get_shape()[0].value == self.rnn_size
            initial_state = tf.reshape(
                tf.tile(initial_state, tf.shape(inputs[0])[:1]),
                [-1, self.rnn_size])

        with tf.variable_scope(scope):

            ## First decoding step
            contexts = [a.attention(initial_state) for a in att_objects]
            output = self._get_rnn_output(inputs[0], initial_state, contexts)
            _, state = cell(tf.concat(1, [inputs[0]] + contexts), initial_state)

            rnn_outputs = [output]
            rnn_states = [initial_state, state]

            for step in range(1, len(inputs)):
                tf.get_variable_scope().reuse_variables()

                if runtime_mode:
                    current_input = self._loop_function(output)
                else:
                    current_input = inputs[step]

                ## N-th decoding step
                contexts = [a.attention(state) for a in att_objects]
                output = self._get_rnn_output(current_input, state, contexts)
                _, state = cell(tf.concat(1, [current_input] + contexts), state)

                rnn_outputs.append(output)
                rnn_states.append(state)

            if summary_collections:
                for i, a in enumerate(att_objects):
                    attentions = a.attentions_in_time[-len(inputs):]
                    alignments = tf.expand_dims(tf.transpose(
                        tf.pack(attentions), perm=[1, 2, 0]), -1)

                    tf.image_summary("attention_{}".format(i), alignments,
                                     collections=summary_collections,
                                     max_images=256)

        return rnn_outputs, rnn_states


    # pylint: disable=no-self-use
    # TODO refactor out of this class to a helper module
    def _get_rnn_output(self, prev_state, prev_output, ctx_tensors):
        """Compute RNN output out of the previous state and output, and the
        context tensors returned from attention mechanisms.

        This function corresponds to the equations for computation the
        t_tilde in the Bahdanau et al. (2015) paper, on page 14,
        **before** the linear projection.

        Arguments:
            prev_state: Previous decoder RNN state. (Denoted s_i-1)
            prev_output: Embedded output of the previous step. (y_i-1)
            ctx_tensors: Context tensors computed by the attentions. (c_i)

        Returns:
            In this decoder, this function just concatenates all the inputs.
        """
        return tf.concat(1, [prev_state, prev_output] + ctx_tensors)


    def _decode(self, rnn_outputs):
        """Decodes a sequence from a list of hidden states

        Arguments:
            rnn_outputs: List of batch x maxout_size tensors
        """
        logits = [self._logit_function(s) for s in rnn_outputs]
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

            ``decoder_{x} for x in range(max_output+2)``
            Training data placeholders. Starts with <s> and ends with </s>

            ``decoder_padding_weights{x} for x in range(max_output+1)``
            Weights used for padding. (Float) tensor of ones and zeros.
            This tensor is one-item shorter than the other one since the
            decoder does not produce the first <s>.

            ``dropout_placeholder``
            Scalar placeholder for dropout probability.
            Has value 'dropout_keep_prob' from the constructor or 1
            in case we are decoding at run-time
        """
        # pylint: disable=invalid-name
        # fd is the common name for feed dictionary
        fd = {}

        start_token_index = self.vocabulary.get_word_index(START_TOKEN)
        fd[self.go_symbols] = np.repeat(start_token_index, len(dataset))

        sentences = dataset.get_series(self.data_id, allow_none=True)

        if sentences is not None:
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences, self.max_output)

            for placeholder, weight in zip(self.train_weights, weights):
                fd[placeholder] = weight

            for placeholder, tensor in zip(self.train_inputs, inputs):
                fd[placeholder] = tensor
        else:
            fd[self.train_inputs[0]] = np.repeat(start_token_index,
                                                 len(dataset))
            for placeholder in self.train_weights:
                fd[placeholder] = np.ones(len(dataset))

        if not train:
            fd[self.dropout_placeholder] = 1.0

        return fd
