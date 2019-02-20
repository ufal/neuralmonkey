"""Implementation of the decoder of the Transformer model.

Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762
"""
# TODO make this code simpler
# pylint: disable=too-many-lines
from typing import Any, Callable, NamedTuple, List, Union, Tuple
import math

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.scaled_dot_product import attention
from neuralmonkey.attention.base_attention import (
    Attendable, get_attention_states, get_attention_mask)
from neuralmonkey.attention.transformer_cross_layer import (
    serial, parallel, flat, hierarchical)
from neuralmonkey.decorators import tensor
from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState, DecoderFeedables, DecoderHistories)
from neuralmonkey.encoders.transformer import (
    TransformerLayer, position_signal)
from neuralmonkey.logging import warn
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import (
    Vocabulary, END_TOKEN_INDEX)
from neuralmonkey.tf_utils import append_tensor, layer_norm


STRATEGIES = ["serial", "parallel", "flat", "hierarchical"]


class TransformerFeedables(NamedTuple(
        "TransformerFeedables",
        [("input_sequence", tf.Tensor),
         ("input_mask", tf.Tensor)])):
    """Additional feedables used only by the Transformer-based decoder.

    Follows the shape pattern of having batch_sized first dimension
    shape(batch_size, ...)

    Attributes:
        input_sequence: The whole input sequence (embedded) that is fed into
            the decoder in each decoding step.
            shape(batch, len, emb)
        input_mask: Mask for masking finished sequences. The last dimension
            is required for compatibility with the beam_search_decoder.
            shape(batch, len, 1)
    """


class TransformerHistories(NamedTuple(
        "TransformerHistories",
        [("self_attention_histories", List[Tuple]),
         ("encoder_attention_histories", List[Tuple])])):
    """The loop state histories specific for Transformer-based decoders.

    Attributes:
        self_attention_histories: A list of ``MultiHeadLoopState`` objects
            populated by values from the self-attention mechanisms used
            in the decoder.
        encoder_attention_histories: A list of ``MultiHeadLoopState`` objects
            populated by values from the encoder-attention mechanisms used
            in the decoder.
    """


# pylint: disable=too-many-instance-attributes
class TransformerDecoder(AutoregressiveDecoder):

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    def __init__(self,
                 name: str,
                 encoders: List[Attendable],
                 vocabulary: Vocabulary,
                 data_id: str,
                 # TODO infer the default for these three from the encoder
                 ff_hidden_size: int,
                 n_heads_self: int,
                 n_heads_enc: Union[List[int], int],
                 depth: int,
                 max_output_len: int,
                 attention_combination_strategy: str = "serial",
                 n_heads_hier: int = None,
                 dropout_keep_prob: float = 1.0,
                 embedding_size: int = None,
                 embeddings_source: EmbeddedSequence = None,
                 tie_embeddings: bool = True,
                 label_smoothing: float = None,
                 self_attention_dropout_keep_prob: float = 1.0,
                 attention_dropout_keep_prob: Union[float, List[float]] = 1.0,
                 use_att_transform_bias: bool = False,
                 supress_unk: bool = False,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Create a decoder of the Transformer model.

        Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762

        Arguments:
            encoders: Input encoders for the decoder.
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
            ff_hidden_size: Size of the feedforward sublayers.
            n_heads_self: Number of the self-attention heads.
            n_heads_enc: Number of the attention heads over each encoder.
                Either a list which size must be equal to ``encoders``, or a
                single integer. In the latter case, the number of heads is
                equal for all encoders.
            attention_comnbination_strategy: One of ``serial``, ``parallel``,
                ``flat``, ``hierarchical``. Controls the attention combination
                strategy for enc-dec attention.
            n_heads_hier: Number of the attention heads for the second
                attention in the ``hierarchical`` attention combination.
            depth: Number of sublayers.
            label_smoothing: A label smoothing parameter for cross entropy
                loss computation.
            attention_dropout_keep_prob: Probability of keeping a value
                during dropout on the attention output.
            supress_unk: If true, decoder will not produce symbols for unknown
                tokens.
            reuse: Reuse the variables from the given model part.
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
            supress_unk=supress_unk,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)

        self.encoders = encoders
        self.ff_hidden_size = ff_hidden_size
        self.n_heads_self = n_heads_self

        if isinstance(n_heads_enc, int):
            if attention_combination_strategy == "flat":
                self.n_heads_enc = [n_heads_enc]
            else:
                self.n_heads_enc = [n_heads_enc for _ in self.encoders]
        else:
            self.n_heads_enc = n_heads_enc

        self.depth = depth
        if isinstance(attention_dropout_keep_prob, float):
            self.attention_dropout_keep_prob = [
                attention_dropout_keep_prob for _ in encoders]
        else:
            self.attention_dropout_keep_prob = attention_dropout_keep_prob
        self.self_att_dropout_keep_prob = self_attention_dropout_keep_prob
        self.use_att_transform_bias = use_att_transform_bias
        self.attention_combination_strategy = attention_combination_strategy
        self.n_heads_hier = n_heads_hier

        self.encoder_states = lambda: [get_attention_states(e)
                                       for e in self.encoders]
        self.encoder_masks = lambda: [get_attention_mask(e)
                                      for e in self.encoders]

        if self.attention_combination_strategy not in STRATEGIES:
            raise ValueError(
                "Unknown attention combination strategy '{}'. "
                "Allowed: {}.".format(self.attention_combination_strategy,
                                      ", ".join(STRATEGIES)))

        if (self.attention_combination_strategy == "hierarchical"
                and self.n_heads_hier is None):
            raise ValueError(
                "You must provide n_heads_hier when using the hierarchical "
                "attention combination strategy.")

        if (self.attention_combination_strategy != "hierarchical"
                and self.n_heads_hier is not None):
            warn("Ignoring n_heads_hier parameter -- use the hierarchical "
                 "attention combination strategy instead.")

        if (self.attention_combination_strategy == "flat"
                and len(self.n_heads_enc) != 1):
            raise ValueError(
                "For the flat attention combination strategy, only a single "
                "value is permitted in n_heads_enc.")

        self._variable_scope.set_initializer(tf.variance_scaling_initializer(
            mode="fan_avg", distribution="uniform"))
    # pylint: enable=too-many-arguments,too-many-locals,too-many-branches

    @property
    def dimension(self) -> int:
        enc_states = self.encoder_states()

        if enc_states:
            first_dim = enc_states[0].get_shape()[2].value  # type: int

            for i, states in enumerate(enc_states):
                enc_dim = states.get_shape()[2].value
                if enc_dim != first_dim:
                    raise ValueError(
                        "Dimension of the {}-th encoder ({}) differs from the "
                        "dimension of the first one ({})."
                        .format(i, enc_dim, first_dim))

            if self.embedding_size is not None:
                if self.embedding_size != first_dim:
                    raise ValueError("Model dimension and input embedding "
                                     "size do not match")
            return first_dim

        if self.embedding_size is None:
            raise ValueError("'embedding_size' must be specified when "
                             "no encoders are provided")

        return self.embedding_size

    @property
    def output_dimension(self) -> int:
        return self.dimension

    def embed_input_symbol(self, inputs: tf.Tensor) -> tf.Tensor:
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)

        if (self.embeddings_source is not None
                and self.embeddings_source.scale_embeddings_by_depth):

            # Pylint @property-related bug
            # pylint: disable=no-member
            embedding_size = self.embedding_matrix.shape.as_list()[-1]
            # pylint: enable=no-member

            embedded *= math.sqrt(embedding_size)

        length = tf.shape(inputs)[1]
        return dropout(embedded + position_signal(self.dimension, length),
                       self.dropout_keep_prob,
                       self.train_mode)

    @tensor
    def train_input_symbols(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)

        # shape (batch, 1 + (time - 1))
        # pylint: disable=unsubscriptable-object
        return tf.concat(
            [tf.expand_dims(self.go_symbols, 1),
             tf.transpose(self.train_inputs[:-1])], 1)
        # pylint: enable=unsubscriptable-object

    def self_attention_sublayer(
            self, prev_layer: TransformerLayer) -> tf.Tensor:
        """Create the decoder self-attention sublayer with output mask."""

        # Layer normalization
        normalized_states = layer_norm(prev_layer.temporal_states)

        # Run self-attention
        # TODO handle attention histories
        self_context, _ = attention(
            queries=normalized_states,
            keys=normalized_states,
            values=normalized_states,
            keys_mask=prev_layer.temporal_mask,
            num_heads=self.n_heads_self,
            masked=True,
            dropout_callback=lambda x: dropout(
                x, self.self_att_dropout_keep_prob, self.train_mode),
            use_bias=self.use_att_transform_bias)

        # Apply dropout
        self_context = dropout(
            self_context, self.dropout_keep_prob, self.train_mode)

        # Add residual connections
        return self_context + prev_layer.temporal_states

    def encoder_attention_sublayer(self, queries: tf.Tensor) -> tf.Tensor:
        """Create the encoder-decoder attention sublayer."""
        enc_states = self.encoder_states()
        enc_masks = self.encoder_masks()
        assert enc_states is not None
        assert enc_masks is not None

        # Attention dropout callbacks are created in a loop so we need to
        # use a factory function to prevent late binding.
        def make_attn_callback(
                prob: float) -> Callable[[tf.Tensor], tf.Tensor]:
            def callback(x: tf.Tensor) -> tf.Tensor:
                return dropout(x, prob, self.train_mode)
            return callback

        dropout_cb = make_attn_callback(self.dropout_keep_prob)
        attn_dropout_cbs = [make_attn_callback(prob)
                            for prob in self.attention_dropout_keep_prob]

        if self.attention_combination_strategy == "serial":
            return serial(queries, enc_states, enc_masks, self.n_heads_enc,
                          attn_dropout_cbs, dropout_cb)

        if self.attention_combination_strategy == "parallel":
            return parallel(queries, enc_states, enc_masks, self.n_heads_enc,
                            attn_dropout_cbs, dropout_cb)

        if self.attention_combination_strategy == "flat":
            assert len(set(self.n_heads_enc)) == 1
            assert len(set(self.attention_dropout_keep_prob)) == 1

            return flat(queries, enc_states, enc_masks, self.n_heads_enc[0],
                        attn_dropout_cbs[0], dropout_cb)

        if self.attention_combination_strategy == "hierarchical":
            assert self.n_heads_hier is not None

            return hierarchical(
                queries, enc_states, enc_masks, self.n_heads_enc,
                self.n_heads_hier, attn_dropout_cbs, dropout_cb)

        raise NotImplementedError(
            "Unknown attention combination strategy: {}"
            .format(self.attention_combination_strategy))

    def feedforward_sublayer(self, layer_input: tf.Tensor) -> tf.Tensor:
        """Create the feed-forward network sublayer."""

        # Layer normalization
        normalized_input = layer_norm(layer_input)

        # Feed-forward network hidden layer + ReLU
        ff_hidden = tf.layers.dense(
            normalized_input, self.ff_hidden_size, activation=tf.nn.relu,
            name="hidden_state")

        # Apply dropout on the activations
        ff_hidden = dropout(ff_hidden, self.dropout_keep_prob, self.train_mode)

        # Feed-forward output projection
        ff_output = tf.layers.dense(ff_hidden, self.dimension, name="output")

        # Apply dropout on the output projection
        ff_output = dropout(ff_output, self.dropout_keep_prob, self.train_mode)

        # Add residual connections
        return ff_output + layer_input

    def layer(self, level: int, inputs: tf.Tensor,
              mask: tf.Tensor) -> TransformerLayer:
        # Recursive implementation. Outputs of the zeroth layer
        # are the inputs

        if level == 0:
            return TransformerLayer(inputs, mask)

        # Compute the outputs of the previous layer
        prev_layer = self.layer(level - 1, inputs, mask)

        with tf.variable_scope("layer_{}".format(level - 1)):

            with tf.variable_scope("self_attention"):
                self_context = self.self_attention_sublayer(prev_layer)

            with tf.variable_scope("encdec_attention"):
                encoder_context = self.encoder_attention_sublayer(self_context)

            with tf.variable_scope("feedforward"):
                output_states = self.feedforward_sublayer(encoder_context)

        # Layer normalization on the decoder output
        if self.depth == level:
            output_states = layer_norm(output_states)

        return TransformerLayer(states=output_states, mask=mask)

    @tensor
    def train_loop_result(self) -> LoopState:
        # We process all decoding the steps together during training.
        # However, we still want to pretend that a proper decoding_loop
        # was called.
        decoder_ls = AutoregressiveDecoder.get_initial_loop_state(self)

        input_sequence = self.embed_input_symbols(self.train_input_symbols)
        input_mask = tf.transpose(self.train_mask)

        last_layer = self.layer(
            self.depth, input_sequence, input_mask)

        tr_feedables = TransformerFeedables(
            input_sequence=input_sequence,
            input_mask=tf.expand_dims(input_mask, -1))

        # t_states shape: (batch, time, channels)
        # dec_w shape: (channels, vocab)
        last_layer_shape = tf.shape(last_layer.temporal_states)
        last_layer_states = tf.reshape(
            last_layer.temporal_states,
            [-1, last_layer_shape[-1]])

        # shape (batch, time, vocab)
        logits = tf.reshape(
            tf.matmul(last_layer_states, self.decoding_w),
            [last_layer_shape[0], last_layer_shape[1], len(self.vocabulary)])
        logits += tf.reshape(self.decoding_b, [1, 1, -1])

        # TODO: record histories properly
        tr_histories = tf.zeros([])
        # tr_histories = TransformerHistories(
        #    self_attention_histories=[
        #        empty_multi_head_loop_state(self.batch_size,
        #                                    self.n_heads_self)
        #        for a in range(self.depth)],
        #    encoder_attention_histories=[
        #        empty_multi_head_loop_state(self.batch_size,
        #                                    self.n_heads_enc)
        #        for a in range(self.depth)])

        feedables = DecoderFeedables(
            step=last_layer_shape[1],
            finished=tf.ones([self.batch_size], dtype=tf.bool),
            embedded_input=self.embed_input_symbols(tf.tile(
                [END_TOKEN_INDEX], [self.batch_size])),
            other=tr_feedables)

        histories = DecoderHistories(
            logits=tf.transpose(logits, perm=[1, 0, 2]),
            output_states=tf.transpose(
                last_layer.temporal_states, [1, 0, 2]),
            output_mask=self.train_mask,
            output_symbols=self.train_inputs,
            other=tr_histories)

        return LoopState(
            feedables=feedables,
            histories=histories,
            constants=decoder_ls.constants)

    def get_initial_loop_state(self) -> LoopState:
        default_ls = AutoregressiveDecoder.get_initial_loop_state(self)
        feedables = default_ls.feedables
        histories = default_ls.histories

        tr_feedables = TransformerFeedables(
            input_sequence=tf.zeros(
                shape=[self.batch_size, 0, self.dimension],
                dtype=tf.float32,
                name="input_sequence"),
            input_mask=tf.zeros(
                shape=[self.batch_size, 0, 1],
                dtype=tf.float32,
                name="input_mask"))

        # TODO: record histories properly
        tr_histories = tf.zeros([])
        # tr_histories = TransformerHistories(
        #    self_attention_histories=[
        #        empty_multi_head_loop_state(self.batch_size,
        #                                    self.n_heads_self)
        #        for a in range(self.depth)],
        #    encoder_attention_histories=[
        #        empty_multi_head_loop_state(self.batch_size,
        #                                    self.n_heads_enc)
        #        for a in range(self.depth)])

        return LoopState(
            histories=histories._replace(other=tr_histories),
            constants=default_ls.constants,
            feedables=feedables._replace(other=tr_feedables))

    def next_state(self, loop_state: LoopState) -> Tuple[tf.Tensor, Any, Any]:
        feedables = loop_state.feedables
        tr_feedables = feedables.other
        tr_histories = loop_state.histories.other

        with tf.variable_scope(self._variable_scope, reuse=tf.AUTO_REUSE):
            # shape (time, batch)
            input_sequence = append_tensor(
                tr_feedables.input_sequence, feedables.embedded_input, 1)

            unfinished_mask = tf.to_float(tf.logical_not(feedables.finished))
            input_mask = append_tensor(
                tr_feedables.input_mask,
                tf.expand_dims(unfinished_mask, -1),
                axis=1)

            last_layer = self.layer(
                self.depth, input_sequence, tf.squeeze(input_mask, -1))

            # (batch, state_size)
            output_state = last_layer.temporal_states[:, -1, :]

        new_feedables = TransformerFeedables(
            input_sequence=input_sequence,
            input_mask=input_mask)

        # TODO: do something more interesting here
        new_histories = tr_histories

        return (output_state, new_feedables, new_histories)
# pylint: enable=too-many-instance-attributes
