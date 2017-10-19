"""Implementation of the decoder of the Transformer model as described in
Vaswani et al. (2017).

See arxiv.org/abs/1706.03762
"""
from typing import Callable, Set, List, Tuple  # pylint: disable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.scaled_dot_product import MultiHeadAttention
from neuralmonkey.decorators import tensor
from neuralmonkey.decoders.sequence_decoder import (
    SequenceDecoder, LoopState, extend_namedtuple, DecoderHistories,
    DecoderFeedables)
from neuralmonkey.encoders.transformer import (
    TransformerLayer, TransformerEncoder, position_signal)
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import (
    Vocabulary, PAD_TOKEN_INDEX, END_TOKEN_INDEX)

# pylint: disable=invalid-name
TransformerHistories = extend_namedtuple(
    "RNNHistories",
    DecoderHistories,
    [("decoded_symbols", tf.TensorArray),
     ("self_attention_histories", List[Tuple]),
     ("inter_attention_histories", List[Tuple])])
# pylint: enable=invalid-name


class TransformerDecoder(SequenceDecoder):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: TransformerEncoder,
                 vocabulary: Vocabulary,
                 data_id: str,
                 # TODO infer the default for these three from the encoder
                 ff_hidden_size: int,
                 n_heads_self: int,
                 n_heads_enc: int,
                 depth: int,
                 max_output_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        check_argument_types()
        SequenceDecoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)

        self.encoder = encoder
        self.ff_hidden_size = ff_hidden_size
        self.n_heads_self = n_heads_self
        self.n_heads_enc = n_heads_enc
        self.depth = depth

        self.dimension = self.encoder.dimension

        self.self_attentions = [None for _ in range(self.depth)] \
            # type: List[MultiHeadAttention]
        self.inter_attentions = [None for _ in range(self.depth)] \
            # type: List[MultiHeadAttention]

        log("Decoder cost op: {}".format(self.cost))
        self._variable_scope.reuse_variables()
        log("Runtime logits: {}".format(self.runtime_logits))
    # pylint: enable=too-many-arguments

    @property
    def output_dimension(self) -> int:
        return self.dimension

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        # TODO better initialization
        return tf.get_variable(
            "word_embeddings", [len(self.vocabulary), self.dimension],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

    def embed_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        length = tf.shape(inputs)[1]
        return embedded + position_signal(self.dimension, length)

    @tensor
    def embedded_train_inputs(self) -> tf.Tensor:
        # THE LAST TRAIN INPUT IS NOT USED IN DECODING FUNCTION
        # (just as a target)
        return self.embed_inputs(
            tf.concat([tf.expand_dims(self.go_symbols, 1),
                       tf.transpose(self.train_inputs[:-1])], 1))

    def get_self_att_object(self, level: int,
                            pr_layer: TransformerLayer) -> MultiHeadAttention:
        if self.self_attentions[level - 1] is None:
            s_ckp = ("dec_self_att_{}_{}".format(level, self._save_checkpoint)
                     if self._save_checkpoint else None)
            l_ckp = ("dec_self_att_{}_{}".format(level, self._load_checkpoint)
                     if self._load_checkpoint else None)

            self.self_attentions[level - 1] = MultiHeadAttention(
                name="mask_self_att_{}".format(level),
                n_heads=self.n_heads_self,
                keys_encoder=pr_layer,
                values_encoder=pr_layer,
                dropout_keep_prob=self.dropout_keep_prob,
                save_checkpoint=s_ckp,
                load_checkpoint=l_ckp)

        return self.self_attentions[level - 1]

    def get_inter_att_object(self, level: int) -> MultiHeadAttention:
        if self.inter_attentions[level - 1] is None:
            s_ckp = ("inter_att_{}_{}".format(level, self._save_checkpoint)
                     if self._save_checkpoint else None)
            l_ckp = ("inter_att_{}_{}".format(level, self._load_checkpoint)
                     if self._load_checkpoint else None)

            self.inter_attentions[level - 1] = MultiHeadAttention(
                name="inter_att_{}".format(level),
                n_heads=self.n_heads_enc,
                keys_encoder=self.encoder,
                values_encoder=self.encoder,
                dropout_keep_prob=self.dropout_keep_prob,
                save_checkpoint=s_ckp,
                load_checkpoint=l_ckp)

        return self.inter_attentions[level - 1]

    def layer(self, level: int, inputs: tf.Tensor,
              mask: tf.Tensor) -> TransformerLayer:

        # Recursive implementation. Outputs of the zeroth layer are the inputs
        if level == 0:
            return TransformerLayer(inputs, mask)

        prev_layer = self.layer(level - 1, inputs, mask)

        # Compute the outputs of this layer

        # TODO generalize att work with 3D queries as default
        with tf.variable_scope("dec_self_att_level_{}".format(level)):
            att = self.get_self_att_object(level, prev_layer)
            self_att_result = att.attention_3d(
                prev_layer.temporal_states, masked=True)

        inter_attention_query = tf.contrib.layers.layer_norm(
            self_att_result + prev_layer.temporal_states)

        # TODO generalize att work with 3D queries as default
        with tf.variable_scope("dec_inter_att_level_{}".format(level)):
            att = self.get_inter_att_object(level)
            inter_att_result = att.attention_3d(inter_attention_query)

        ff_input = tf.contrib.layers.layer_norm(
            inter_att_result + inter_attention_query)

        ff_hidden = tf.layers.dense(ff_input, self.ff_hidden_size,
                                    activation=tf.nn.relu,
                                    name="ff_hidden_{}".format(level))

        ff_output = tf.layers.dense(ff_hidden, self.dimension,
                                    name="ff_out_{}".format(level))

        output_states = tf.contrib.layers.layer_norm(ff_output + ff_input)
        return TransformerLayer(states=output_states, mask=mask)

    @tensor
    def train_logits(self) -> tf.Tensor:
        last_layer = self.layer(self.depth, self.embedded_train_inputs,
                                tf.transpose(self.train_mask))

        temporal_states = dropout(last_layer.temporal_states,
                                  self.dropout_keep_prob, self.train_mode)

        # matmul with output matrix
        # t_states shape: (batch, time, channels)
        # dec_w shape: (channels, vocab)

        logits = tf.nn.conv1d(
            temporal_states, tf.expand_dims(self.decoding_w, 0), 1, "SAME")

        return tf.transpose(
            logits + tf.expand_dims(tf.expand_dims(self.decoding_b, 0), 0),
            perm=[1, 0, 2])

    def get_initial_loop_state(self) -> LoopState:

        default_ls = SequenceDecoder.get_initial_loop_state(self)
        # feedables = default_ls.feedables._asdict()
        histories = default_ls.histories._asdict()

        histories["self_attention_histories"] = [
            a.initial_loop_state() for a in self.self_attentions]

        histories["inter_attention_histories"] = [
            a.initial_loop_state() for a in self.inter_attentions]

        histories["decoded_symbols"] = tf.TensorArray(
            dtype=tf.int32, dynamic_size=True, size=0,
            clear_after_read=False, name="decoded_symbols")

        tr_histories = TransformerHistories(**histories)

        return LoopState(
            histories=tr_histories,
            constants=default_ls.constants,
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

            # shape (batch, time, dimension)
            embedded_inputs = self.embed_inputs(tf.transpose(decoded_symbols))

            # MASKA (time, batch)
            mask = tf.to_float(histories.mask.stack())
            mask.set_shape([None, None])

            last_layer = self.layer(self.depth, embedded_inputs,
                                    tf.transpose(mask))

            # (batch, state_size)
            output_state = tf.transpose(
                last_layer.temporal_states, perm=[1, 0, 2])[-1]

            logits = tf.matmul(output_state, self.decoding_w) + self.decoding_b

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
            has_finished = tf.logical_or(feedables.finished, has_just_finished)
            not_finished = tf.logical_not(has_finished)

            new_feedables = DecoderFeedables(
                step=step + 1,
                finished=has_finished,
                input_symbol=next_symbols,
                prev_logits=logits)

            new_histories = TransformerHistories(
                logits=histories.logits.write(step, logits),
                decoder_outputs=histories.decoder_outputs.write(
                    step, output_state),
                mask=histories.mask.write(step, not_finished),
                # transformer-specific:
                decoded_symbols=decoded_symbols_ta,
                self_attention_histories=histories.self_attention_histories,
                inter_attention_histories=histories.inter_attention_histories)

            new_loop_state = LoopState(
                histories=new_histories,
                constants=loop_state.constants,
                feedables=new_feedables)

            return new_loop_state
        # pylint: enable=too-many-locals

        return body

    def get_dependencies(self) -> Set[ModelPart]:
        default_dependencies = ModelPart.get_dependencies(self)

        assert all(self.self_attentions)
        assert all(self.inter_attentions)

        dependencies = set(self.self_attentions + self.inter_attentions)
        return default_dependencies.union(dependencies)
