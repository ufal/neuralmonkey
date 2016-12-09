#tests: mypy, lint

from typing import List, Any, Callable, Optional
import math

import tensorflow as tf
import numpy as np

from neuralmonkey.vocabulary import Vocabulary, START_TOKEN
from neuralmonkey.logging import log
from neuralmonkey.nn.utils import dropout
from neuralmonkey.decoders.encoder_projection import (
    linear_encoder_projection, concat_encoder_projection, empty_initial_state)


class Decoder(object):

    def __init__(self,
                 encoders: List[object],
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_output_len: int,
                 dropout_keep_prob: float,
                 rnn_size: Optional[int]=None,
                 embedding_size: Optional[int]=None,
                 output_projection: Optional[Callable[
                     [tf.Tensor, tf.Tensor, List[tf.Tensor]], tf.Tensor]]=None,
                 encoder_projection: Optional[Callable[
                     [tf.Tensor, Optional[int], Optional[List[object]]],
                     tf.Tensor]]=None,
                 use_attention: bool=False,
                 embeddings_encoder: Optional[object]=None):
        """Creates a refactored version of monster decoder.

        Arguments:
            there are many argumetns.
        """
        log("Initializing decoder, name: '{}'".format(name))

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.name = name
        self.max_output_len = max_output_len
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.output_projection = output_projection
        self.encoder_projection = encoder_projection
        self.use_attention = use_attention
        self.embeddings_encoder = embeddings_encoder

        if self.embedding_size is None and self.embeddings_encoder is None:
            raise ValueError("You must specify either embedding size or the "
                             "encoder from which to reuse the embeddings ("
                             "e.g. set either 'embedding_size' or "
                             " 'embeddings_encoder' parameter)")

        if self.embeddings_encoder is not None:
            if self.embedding_size is not None:
                log("Warning: Overriding the embedding_size parameter with the "
                    "size of the reused embeddings from the encoder.",
                    color="red")

            self.embedding_size = (
                self.embeddings_encoder.embedding_matrix.get_shape()[1].value)

        if self.encoder_projection is None:
            if len(self.encoders) == 0:
                log("No encoder - language model only.")
                self.encoder_projection = empty_initial_state
            elif rnn_size is None:
                log("No rnn_size or encoder_projection: Using concatenation of "
                    "encoded states")
                self.encoder_projection = concat_encoder_projection
            else:
                log("Using linear projection of encoders as the initial state")
                self.encoder_projection = linear_encoder_projection(
                    self.dropout_keep_prob)

        with tf.variable_scope(name):
            self._create_input_placeholders()
            self._create_training_placeholders()
            self._create_initial_state()
            self._create_embedding_matrix()

            self.decoding_w = tf.Variable(
                tf.random_uniform([rnn_size, len(vocabulary)], -0.5, 0.5),
                name="state_to_word_W")

            self.decoding_b = tf.Variable(
                tf.fill([len(vocabulary)], - math.log(len(vocabulary))),
                name="state_to_word_b")

            embedded_train_inputs = [self._embed_and_dropout(o)
                                     for o in self.train_inputs[:-1]]

            def loop(prev_state):
                out_activation = self._logit_function(prev_state)
                prev_word_index = tf.argmax(out_activation, 1)
                return self._embed_and_dropout(prev_word_index)

            train_rnn_outputs, _ = self._attention_decoder(
                embedded_train_inputs, self.initial_state,
                self.embedding_size)

            tf.get_variable_scope().reuse_variables()

            runtime_inputs = [tf.nn.embedding_lookup(self.embedding_matrix,
                                                     self.go_symbols)]
            runtime_inputs += [None for _ in range(self.max_output_len)]

            ### POZOR TADY SE NEDELA DROPOUT

            runtime_rnn_outputs, _ = self._attention_decoder(
                runtime_inputs, self.initial_state, self.embedding_size,
                loop_function=loop)

            self.hidden_states = runtime_rnn_outputs

        def decode(rnn_outputs):
            logits = []
            decoded = []

            for out in rnn_outputs:
                out_activation = self._logit_function(out)
                logits.append(out_activation)
                decoded.append(tf.argmax(out_activation[:, 1:], 1) + 1)

            return decoded, logits


        _, self.train_logits = decode(train_rnn_outputs)

        self.train_loss = tf.nn.seq2seq.sequence_loss(
            self.train_logits, self.train_targets, self.train_padding,
            len(self.vocabulary))

        self.cost = self.train_loss

        self.decoded, self.runtime_logits = decode(runtime_rnn_outputs)

        self.runtime_loss = tf.nn.seq2seq.sequence_loss(
            self.runtime_logits, self.train_targets, self.train_padding,
            len(self.vocabulary))

        self.runtime_logprobs = [tf.nn.log_softmax(l)
                                 for l in self.runtime_logits]

        tf.scalar_summary('train_loss_with_gt_intpus',
                          self.train_loss,
                          collections=["summary_train"])

        tf.scalar_summary('train_loss_with_decoded_inputs',
                          self.runtime_loss,
                          collections=["summary_train"])

        tf.scalar_summary('train_optimization_cost', self.cost,
                          collections=["summary_train"])

        log("Decoder initalized.")



    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, name="decoder_train_mode")

        self.go_symbols = tf.placeholder(tf.int32, shape=[None],
                                         name="decoder_go_symbols")


    def _create_training_placeholders(self):
        """Creates training placeholder nodes in the computation graph

        The training placeholder nodes are NOT fed during runtime.
        """
        self.train_inputs = []
        with tf.variable_scope("decoder_inputs"):
            for i in range(self.max_output_len + 2):
                dec = tf.placeholder(
                    tf.int64, [None], name='decoder{0}'.format(i))
                tf.add_to_collection('dec_encoder_ins', dec)
                self.train_inputs.append(dec)

        self.train_targets = self.train_inputs[1:]

        self.train_padding = []
        with tf.variable_scope("input_weights"):
            for _ in range(len(self.train_targets)):
                self.train_padding.append(tf.placeholder(tf.float32, [None]))

    def _create_initial_state(self):
        """Construct the part of the computation graph that computes the initial
        state of the decoder."""
        self.initial_state = dropout(self.encoder_projection(self.train_mode,
                                                             self.rnn_size,
                                                             self.encoders),
                                     self.dropout_keep_prob,
                                     self.train_mode)
        # TODO broadcast if initial state is 1D tensor
        # (move from attention_decoder)

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        if self.embeddings_encoder is None:
            # TODO better initialization
            self.embedding_matrix = tf.Variable(
                tf.random_uniform([len(self.vocabulary), self.embedding_size],
                                  -0.5, 0.5),
                name="word_embeddings")
        else:
            self.embedding_matrix = self.embeddings_encoder.embedding_matrix

    def _embed_and_dropout(self, inputs):
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return dropout(embedded, self.dropout_keep_prob, self.train_mode)


    def _logit_function(self, state):
        state = dropout(state, self.dropout_keep_prob, self.train_mode)
        return tf.matmul(state, self.decoding_w) + self.decoding_b




    def _attention_decoder(self, decoder_inputs, initial_state,
                           embedding_size, output_size=None,
                           loop_function=None, dtype=tf.float32, scope=None):

        cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)

        att_objects = []
        if self.use_attention:
            att_objects = [e.attention_object for e in self.encoders
                           if e.attention_object is not None]


        if output_size is None:
            output_size = cell.output_size

        with tf.variable_scope(scope or "attention_decoder"):
            batch_size = tf.shape(decoder_inputs[0])[0]    # Needed for reshaping.

            # do manualy broadcasting of the initial state if we want it
            # to be the same for all inputs
            if len(initial_state.get_shape()) == 1:
                state_size = initial_state.get_shape()[0].value
                initial_state = tf.reshape(tf.tile(initial_state,
                                                   tf.shape(decoder_inputs[0])[:1]),
                                           [-1, state_size])

            state = initial_state
            outputs = []
            prev = None

            def initialize(attention_obj, batch_size, dtype):
                batch_attn_size = tf.pack([batch_size, attention_obj.attn_size])
                initial = tf.zeros(batch_attn_size, dtype=dtype)
                # Ensure the second shape of attention vectors is set.
                initial.set_shape([None, attention_obj.attn_size])
                return initial

            attns = [initialize(a, batch_size, dtype) for a in att_objects]

            states = []
            for i, inp in enumerate(decoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev)
                # Merge input and previous attentions into one vector of the
                # right size.
                x = tf.nn.seq2seq.linear([inp] + attns, embedding_size, True)
                # Run the RNN.

                cell_output, state = cell(x, state)
                states.append(state)
                # Run the attention mechanism.
                attns = [a.attention(state) for a in att_objects]

                if attns:
                    with tf.variable_scope("AttnOutputProjection"):
                        output = tf.nn.seq2seq.linear(
                            [cell_output] + attns, output_size, True)
                else:
                    output = cell_output

                if loop_function is not None:
                    prev = output
                outputs.append(output)

        return outputs, states







    def feed_dict(self, dataset, train=False):
        sentences = dataset.get_series(self.data_id, allow_none=True)
        res = {}

        start_token_index = self.vocabulary.get_word_index(START_TOKEN)
        res[self.go_symbols] = np.repeat(start_token_index, len(dataset))
        res[self.train_mode] = train

        if sentences is not None:
            sentnces_tensors, weights_tensors = \
                self.vocabulary.sentences_to_tensor(sentences,
                                                    self.max_output_len)

            for weight_plc, weight_tensor in zip(self.train_padding,
                                                 weights_tensors):
                res[weight_plc] = weight_tensor

            for words_plc, words_tensor in zip(self.train_inputs,
                                               sentnces_tensors):
                res[words_plc] = words_tensor

        return res
