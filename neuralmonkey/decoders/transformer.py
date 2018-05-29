"""Implementation of the decoder of the Transformer model.

Described in Vaswani et al. (2017), arxiv.org/abs/1706.03762
"""
# TODO make this code simpler
# pylint: disable=too-many-lines
# pylint: disable=unused-import
from typing import Callable, Set, List, Tuple, Union
# pylint: enable=unused-import

import math

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
from neuralmonkey.logging import log, warn
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import (
    Vocabulary, PAD_TOKEN_INDEX, END_TOKEN_INDEX)
from neuralmonkey.tf_utils import layer_norm

# pylint: disable=invalid-name
TransformerHistories = extend_namedtuple(
    "RNNHistories",
    DecoderHistories,
    [("decoded_symbols", tf.TensorArray),
     ("self_attention_histories", List[Tuple]),
     ("inter_attention_histories", List[Tuple]),
     ("input_mask", tf.TensorArray)])
# pylint: enable=invalid-name

STRATEGIES = ["serial", "parallel", "flat", "hierarchical"]


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
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
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

        Keyword arguments:
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

        self.encoder_states = [get_attention_states(e) for e in self.encoders]
        self.encoder_masks = [get_attention_mask(e) for e in self.encoders]
        self.dimension = self.encoder_states[0].get_shape()[2].value

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

        for i, enc_states in enumerate(self.encoder_states):
            enc_dim = enc_states.get_shape()[2].value
            if enc_dim != self.dimension:
                raise ValueError("Dimension of the {}-th encoder ({}) differs "
                                 "from the dimension of the first one ({})."
                                 .format(i, enc_dim, self.dimension))

        if self.embedding_size != self.dimension:
            raise ValueError("Model dimension and input embedding size"
                             "do not match")

        self._variable_scope.set_initializer(tf.variance_scaling_initializer(
            mode="fan_avg", distribution="uniform"))

        log("Decoder cost op: {}".format(self.cost))
        self._variable_scope.reuse_variables()
        log("Runtime logits: {}".format(self.runtime_logits))
    # pylint: enable=too-many-arguments,too-many-locals,too-many-branches

    @property
    def output_dimension(self) -> int:
        return self.dimension

    def embed_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)

        if (self.embeddings_source is not None
                and self.embeddings_source.scale_embeddings_by_depth):

            # Pylint @property-related bug
            # pylint: disable=no-member
            embedding_size = self.embedding_matrix.shape.as_list()[-1]
            # pylint: enable=no-member

            embedded *= math.sqrt(embedding_size)

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

    def self_attention_sublayer(
            self, prev_layer: TransformerLayer) -> tf.Tensor:
        """Create the decoder self-attention sublayer with output mask."""

        # Layer normalization
        normalized_states = layer_norm(prev_layer.temporal_states)

        # Run self-attention
        # TODO handle histories
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

    # pylint: disable=too-many-arguments
    def generic_encatt_sublayer(
            self,
            queries: tf.Tensor,
            states: tf.Tensor,
            mask: tf.Tensor,
            n_heads: int,
            attention_dropout_keep_prob: float,
            normalize: bool = True,
            use_dropout: bool = True,
            residual: bool = True) -> tf.Tensor:
        # Layer normalization
        if normalize:
            normalized_queries = layer_norm(queries)
        else:
            normalized_queries = queries

        # Attend to the encoder
        # TODO handle histories
        encoder_context, _ = attention(
            queries=normalized_queries,
            keys=states,
            values=states,
            keys_mask=mask,
            num_heads=n_heads,
            dropout_callback=lambda x: dropout(
                x, attention_dropout_keep_prob, self.train_mode),
            use_bias=self.use_att_transform_bias)

        # Apply dropout
        if use_dropout:
            encoder_context = dropout(
                encoder_context, self.dropout_keep_prob, self.train_mode)

        # Add residual connections
        if residual:
            encoder_context += queries

        return encoder_context
    # pylint: enable=too-many-arguments

    # pylint: disable=too-many-locals
    def encoder_attention_sublayer(self, queries: tf.Tensor) -> tf.Tensor:
        """Create the encoder-decoder attention sublayer."""

        # serial:
        # * repeat for every encoder:
        #   - lnorm + attend + dropout + add residual
        # * update queiies between layers
        if self.attention_combination_strategy == "serial":
            for i, (heads, states, mask, drop_prob) in enumerate(zip(
                    self.n_heads_enc, self.encoder_states,
                    self.encoder_masks, self.attention_dropout_keep_prob)):

                with tf.variable_scope("enc_{}".format(i)):
                    queries = self.generic_encatt_sublayer(
                        queries, states, mask, heads, drop_prob)

            encoder_context = queries

        # parallel:
        # * normalize queries,
        # * attend and dropout independently for every encoder,
        # * sum up the results
        # * add residual and return
        if self.attention_combination_strategy == "parallel":
            normalized_queries = layer_norm(queries)
            contexts = []

            for i, (heads, states, mask, drop_prob) in enumerate(zip(
                    self.n_heads_enc, self.encoder_states,
                    self.encoder_masks, self.attention_dropout_keep_prob)):

                with tf.variable_scope("enc_{}".format(i)):
                    contexts.append(
                        self.generic_encatt_sublayer(
                            normalized_queries, states, mask, heads,
                            drop_prob, normalize=False, residual=False))

            encoder_context = sum(contexts) + queries

        # hierarachical:
        # * normalize queries
        # * attend to every encoder
        # * attend to the resulting context vectors (reuse normalized queries)
        # * apply dropout, add residual connection and return
        if self.attention_combination_strategy == "hierarchical":
            normalized_queries = layer_norm(queries)
            contexts = []

            batch = tf.shape(queries)[0]
            time_q = tf.shape(queries)[1]

            for i, (heads, states, mask, drop_prob) in enumerate(zip(
                    self.n_heads_enc, self.encoder_states,
                    self.encoder_masks, self.attention_dropout_keep_prob)):

                with tf.variable_scope("enc_{}".format(i)):
                    contexts.append(
                        self.generic_encatt_sublayer(
                            normalized_queries, states, mask, heads,
                            drop_prob, normalize=False, residual=False))

            # context is of shape [batch, time(q), channels(v)],
            # stack to [batch, time(q), n_encoders, channels(v)]
            # reshape to [batch x time(q), n_encoders, channels(v)]
            stacked_contexts = tf.reshape(
                tf.stack(contexts, axis=2),
                [batch * time_q, len(self.encoders), self.dimension])

            # hierarchical mask: ones of shape [batch x time(q), n_encoders]
            hier_mask = tf.ones([batch * time_q, len(self.encoders)])

            # reshape queries to [batch x time(q), 1, channels(v)]
            reshaped_queries = tf.reshape(
                normalized_queries, [batch * time_q, 1, self.dimension])

            # returned shape [batch x time(q), 1, channels(v)]
            with tf.variable_scope("enc_hier"):
                assert self.n_heads_hier is not None
                encoder_context_stacked_batch = self.generic_encatt_sublayer(
                    reshaped_queries, stacked_contexts, hier_mask,
                    self.n_heads_hier, self.dropout_keep_prob,
                    normalize=False, use_dropout=False,
                    residual=False)

            # reshape back to [batch, time(q), channels(v)]
            encoder_context = tf.reshape(
                encoder_context_stacked_batch, [batch, time_q, self.dimension])

            encoder_context = dropout(
                encoder_context, self.dropout_keep_prob, self.train_mode)

            encoder_context += queries

        # flat:
        # * concatenate the states and mask along the time axis
        # * run attention over the concatenation
        if self.attention_combination_strategy == "flat":

            concat_mask = tf.concat(self.encoder_masks, 1)
            concat_states = tf.concat(self.encoder_states, 1)

            encoder_context = self.generic_encatt_sublayer(
                queries, concat_states, concat_mask, self.n_heads_enc[0],
                self.attention_dropout_keep_prob[0])

        return encoder_context
    # pylint: enable=too-many-locals

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
    def train_logits(self) -> tf.Tensor:
        last_layer = self.layer(self.depth, self.embedded_train_inputs,
                                tf.transpose(self.train_mask))

        # t_states shape: (batch, time, channels)
        # dec_w shape: (channels, vocab)
        last_layer_shape = tf.shape(last_layer.temporal_states)
        last_layer_states = tf.reshape(
            last_layer.temporal_states,
            [-1, last_layer_shape[-1]])

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
            empty_multi_head_loop_state(self.n_heads_enc[0])  # TODO all heads
            for a in range(self.depth)]

        histories["decoded_symbols"] = tf.TensorArray(
            dtype=tf.int32, dynamic_size=True, size=0,
            clear_after_read=False, name="decoded_symbols")

        histories["input_mask"] = tf.TensorArray(
            dtype=tf.float32, dynamic_size=True, size=0,
            clear_after_read=False, name="input_mask")

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

            input_mask = histories.input_mask.write(
                step, tf.to_float(tf.logical_not(feedables.finished)))

            # shape (time, batch)
            decoded_symbols = decoded_symbols_ta.stack()
            decoded_symbols.set_shape([None, None])
            decoded_symbols_in_batch = tf.transpose(decoded_symbols)

            # mask (time, batch)
            mask = input_mask.stack()
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
                input_mask=input_mask)
            # pylint: enable=not-callable

            new_loop_state = LoopState(
                histories=new_histories,
                constants=[],
                feedables=new_feedables)

            return new_loop_state
        # pylint: enable=too-many-locals

        return body
# pylint: enable=too-many-instance-attributes
