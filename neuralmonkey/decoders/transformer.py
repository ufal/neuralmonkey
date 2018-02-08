"""Implementation of the decoder of the Transformer model.

Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762
"""
from typing import Callable, Set, List, Tuple  # pylint: disable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.scaled_dot_product import (
    attention, empty_multi_head_loop_state)
from neuralmonkey.attention.base_attention import (
    Attendable, get_attention_states, get_attention_mask)
from neuralmonkey.decorators import tensor
from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState, extend_namedtuple, DecoderHistories,
    DecoderFeedables)
from neuralmonkey.encoders.transformer import (
    TransformerLayer, position_signal)
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.logging import log
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import (
    Vocabulary, PAD_TOKEN_INDEX, END_TOKEN_INDEX)

# pylint: disable=invalid-name
TransformerHistories = extend_namedtuple(
    "RNNHistories",
    DecoderHistories,
    [("decoded_symbols", tf.TensorArray),
     ("self_attention_histories", List[Tuple]),
     ("inter_attention_histories", List[Tuple]),
     ("input_mask", tf.TensorArray)])
# pylint: enable=invalid-name


class TransformerDecoder(AutoregressiveDecoder):

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 encoder: Attendable,
                 vocabulary: Vocabulary,
                 data_id: str,
                 # TODO infer the default for these three from the encoder
                 ff_hidden_size: int,
                 n_heads_self: int,
                 n_heads_enc: int,
                 depth: int,
                 max_output_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 embedding_size: int = None,
                 embeddings_source: EmbeddedSequence = None,
                 tie_embeddings: bool = True,
                 label_smoothing: float = None,
                 attention_dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a decoder of the Transformer model.

        Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762

        Arguments:
            encoder: Input encoder of the decoder.
            vocabulary: Target vocabulary.
            data_id: Target data series.
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            max_output_len: Maximum length of an output sequence.
            dropout_keep_prob: Probability of keeping a value during dropout.
            embedding_size: Size of embedding vectors for target words.
            embeddings_source: Embedded sequence to take embeddings from.
            tie_embeddings: Use decoder.embedding_matrix also in place
                of the output decoding matrix.

        Keyword arguments:
            ff_hidden_size: Size of the feedforward sublayers.
            n_heads_self: Number of the self-attention heads.
            n_heads_enc: Number of the attention heads over the encoder.
            depth: Number of sublayers.
            label_smoothing: A label smoothing parameter for cross entropy
                loss computation.
            attention_dropout_keep_prob: Probability of keeping a value
                during dropout on the attention output.
        """
        check_argument_types()
        AutoregressiveDecoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            embedding_size=embedding_size,
            embeddings_source=embeddings_source,
            tie_embeddings=tie_embeddings,
            label_smoothing=label_smoothing,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)

        self.encoder = encoder
        self.ff_hidden_size = ff_hidden_size
        self.n_heads_self = n_heads_self
        self.n_heads_enc = n_heads_enc
        self.depth = depth
        self.attention_dropout_keep_prob = attention_dropout_keep_prob

        self.encoder_states = get_attention_states(self.encoder)
        self.encoder_mask = get_attention_mask(self.encoder)
        self.dimension = self.encoder_states.get_shape()[2].value

        if self.embedding_size != self.dimension:
            raise ValueError("Model dimension and input embedding size"
                             "do not match")

        log("Decoder cost op: {}".format(self.cost))
        self._variable_scope.reuse_variables()
        log("Runtime logits: {}".format(self.runtime_logits))
    # pylint: enable=too-many-arguments,too-many-locals

    @property
    def output_dimension(self) -> int:
        return self.dimension

    def embed_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        length = tf.shape(inputs)[1]
        return embedded + position_signal(self.dimension, length)

    @tensor
    def embedded_train_inputs(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)

        # shape (batch, 1 + (time - 1))
        input_tokens = tf.concat(
            [tf.expand_dims(self.go_symbols, 1),
             tf.transpose(self.train_inputs[:-1])], 1)

        input_embeddings = self.embed_inputs(input_tokens)

        return dropout(input_embeddings,
                       self.dropout_keep_prob,
                       self.train_mode)

    def masked_self_attention(
            self, level: int, prev_layer: TransformerLayer) -> tf.Tensor:

        with tf.variable_scope("dec_self_att_level_{}".format(level),
                               reuse=tf.AUTO_REUSE):
            # TODO handle histories
            self_context, _ = attention(
                queries=prev_layer.temporal_states,
                keys=prev_layer.temporal_states,
                values=prev_layer.temporal_states,
                keys_mask=prev_layer.temporal_mask,
                num_heads=self.n_heads_self,
                masked=True,
                dropout_callback=lambda x: dropout(
                    x, self.attention_dropout_keep_prob, self.train_mode))

            return dropout(
                self_context, self.dropout_keep_prob, self.train_mode)

    def encoder_attention(self, level: int, queries: tf.Tensor) -> tf.Tensor:

        with tf.variable_scope("dec_inter_att_level_{}".format(level),
                               reuse=tf.AUTO_REUSE):
            encoder_att_states = get_attention_states(self.encoder)
            encoder_att_mask = get_attention_mask(self.encoder)

            # TODO handle histories
            encoder_context, _ = attention(
                queries=queries,
                keys=encoder_att_states,
                values=encoder_att_states,
                keys_mask=encoder_att_mask,
                num_heads=self.n_heads_enc,
                dropout_callback=lambda x: dropout(
                    x, self.attention_dropout_keep_prob, self.train_mode))

            return dropout(
                encoder_context, self.dropout_keep_prob, self.train_mode)

    def layer(self, level: int, inputs: tf.Tensor,
              mask: tf.Tensor) -> TransformerLayer:
        # Recursive implementation. Outputs of the zeroth layer are the inputs
        if level == 0:
            norm_inputs = tf.contrib.layers.layer_norm(
                inputs, begin_norm_axis=2)
            return TransformerLayer(norm_inputs, mask)

        # Compute the outputs of the previous layer
        prev_layer = self.layer(level - 1, inputs, mask)

        # Run self-attention
        self_context = self.masked_self_attention(level, prev_layer)

        # Residual connections + layer normalization
        encoder_queries = tf.contrib.layers.layer_norm(
            self_context + prev_layer.temporal_states, begin_norm_axis=2)

        # Attend to the encoder
        encoder_context = self.encoder_attention(level, encoder_queries)

        # Residual connections + layer normalization
        ff_input = tf.contrib.layers.layer_norm(
            encoder_context + encoder_queries, begin_norm_axis=2)

        # Feed-forward network hidden layer + ReLU + dropout
        ff_hidden = tf.layers.dense(
            ff_input, self.ff_hidden_size, activation=tf.nn.relu,
            name="ff_hidden_{}".format(level))
        ff_hidden_drop = dropout(
            ff_hidden, self.dropout_keep_prob, self.train_mode)

        # Feed-forward output projection + dropout
        ff_output = tf.layers.dense(
            ff_hidden_drop, self.dimension, name="ff_out_{}".format(level))
        ff_output = dropout(ff_output, self.dropout_keep_prob, self.train_mode)

        # Residual connections + layer normalization
        output_states = tf.contrib.layers.layer_norm(
            ff_output + ff_input, begin_norm_axis=2)

        return TransformerLayer(states=output_states, mask=mask)

    @tensor
    def train_logits(self) -> tf.Tensor:
        last_layer = self.layer(self.depth, self.embedded_train_inputs,
                                tf.transpose(self.train_mask))
        # t_states shape: (batch, time, channels)
        # dec_w shape: (channels, vocab)
        last_layer_shape = tf.shape(last_layer.temporal_states)
        last_layer_states = tf.reshape(
            last_layer.temporal_states,
            [-1, last_layer_shape[-1]])

        # TODO: Add bias after matmul by embedding_matrix?

        # Reusing input embedding matrix for generating logits
        # significantly reduces the overall size of the model.
        # See: https://arxiv.org/pdf/1608.05859.pdf
        #
        # shape (batch, time, vocab)
        logits = tf.reshape(
            tf.matmul(last_layer_states, self.decoding_w),
            [last_layer_shape[0], last_layer_shape[1], len(self.vocabulary)])
        logits += tf.reshape(self.decoding_b, [1, 1, -1])

        # return logits in time-major shape
        return tf.transpose(logits, perm=[1, 0, 2])

    def get_initial_loop_state(self) -> LoopState:

        default_ls = AutoregressiveDecoder.get_initial_loop_state(self)
        # feedables = default_ls.feedables._asdict()
        histories = default_ls.histories._asdict()

        histories["self_attention_histories"] = [
            empty_multi_head_loop_state(self.n_heads_self)
            for a in range(self.depth)]

        histories["inter_attention_histories"] = [
            empty_multi_head_loop_state(self.n_heads_enc)
            for a in range(self.depth)]

        histories["decoded_symbols"] = tf.TensorArray(
            dtype=tf.int32, dynamic_size=True, size=0,
            clear_after_read=False, name="decoded_symbols")

        input_mask = tf.TensorArray(
            dtype=tf.float32, dynamic_size=True, size=0,
            clear_after_read=False, name="input_mask")

        histories["input_mask"] = input_mask.write(
            0, tf.ones_like(self.go_symbols, dtype=tf.float32))

        # TransformerHistories is a type and should be callable
        # pylint: disable=not-callable
        tr_histories = TransformerHistories(**histories)
        # pylint: enable=not-callable

        return LoopState(
            histories=tr_histories,
            constants=[],
            feedables=default_ls.feedables)

    def get_body(self, train_mode: bool, sample: bool = False) -> Callable:
        assert not train_mode

        # pylint: disable=too-many-locals
        def body(*args) -> LoopState:

            loop_state = LoopState(*args)
            histories = loop_state.histories
            feedables = loop_state.feedables
            step = feedables.step

            decoded_symbols_ta = histories.decoded_symbols.write(
                step, feedables.input_symbol)

            # shape (time, batch)
            decoded_symbols = decoded_symbols_ta.stack()
            decoded_symbols.set_shape([None, None])
            decoded_symbols_in_batch = tf.transpose(decoded_symbols)

            # MASKA (time, batch)
            mask = histories.input_mask.stack()
            mask.set_shape([None, None])

            with tf.variable_scope(self._variable_scope, reuse=tf.AUTO_REUSE):

                # shape (batch, time, dimension)
                embedded_inputs = self.embed_inputs(decoded_symbols_in_batch)

                last_layer = self.layer(
                    self.depth, embedded_inputs, tf.transpose(mask))

                # (batch, state_size)
                output_state = last_layer.temporal_states[:, -1, :]

                # See train_logits definition
                logits = tf.matmul(output_state, self.decoding_w)
                logits += self.decoding_b

                if sample:
                    next_symbols = tf.multinomial(logits, num_samples=1)
                else:
                    next_symbols = tf.to_int32(tf.argmax(logits, axis=1))
                    int_unfinished_mask = tf.to_int32(
                        tf.logical_not(loop_state.feedables.finished))

                    # Note this works only when PAD_TOKEN_INDEX is 0. Otherwise
                    # this have to be rewritten
                    assert PAD_TOKEN_INDEX == 0
                    next_symbols = next_symbols * int_unfinished_mask

                    has_just_finished = tf.equal(next_symbols, END_TOKEN_INDEX)
                    has_finished = tf.logical_or(feedables.finished,
                                                 has_just_finished)
                    not_finished = tf.logical_not(has_finished)

            new_feedables = DecoderFeedables(
                step=step + 1,
                finished=has_finished,
                input_symbol=next_symbols,
                prev_logits=logits)

            # TransformerHistories is a type and should be callable
            # pylint: disable=not-callable
            new_histories = TransformerHistories(
                logits=histories.logits.write(step, logits),
                decoder_outputs=histories.decoder_outputs.write(
                    step, output_state),
                mask=histories.mask.write(step, not_finished),
                outputs=histories.outputs.write(step, next_symbols),
                # transformer-specific:
                # TODO handle attention histories correctly
                decoded_symbols=decoded_symbols_ta,
                self_attention_histories=histories.self_attention_histories,
                inter_attention_histories=histories.inter_attention_histories,
                input_mask=histories.input_mask.write(
                    step + 1, tf.to_float(not_finished)))
            # pylint: enable=not-callable

            new_loop_state = LoopState(
                histories=new_histories,
                constants=[],
                feedables=new_feedables)

            return new_loop_state
        # pylint: enable=too-many-locals

        return body
