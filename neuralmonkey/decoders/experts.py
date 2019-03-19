from typing import NamedTuple, Callable, List, Optional

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState, DecoderFeedables, DecoderHistories)
from neuralmonkey.decorators import tensor
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.tf_utils import append_tensor, get_variable, get_shape_list
from neuralmonkey.vocabulary import (
    Vocabulary, END_TOKEN_INDEX, PAD_TOKEN_INDEX)


STRATEGIES = ["uniform", "global", "context-aware"]


class ExpertFeedables(NamedTuple(
        "ExpertFeedables",
        [("dec_loop_states", List[LoopState])])):
    """TODO

    """


class ExpertHistories(NamedTuple(
        "ExpertHistories",
        [("probs", tf.Tensor),
         ("weights", tf.Tensor)])):
    """TODO

    """


class ExpertDecoder(AutoregressiveDecoder):
    """In-graph mixture of experts decocer.

    TODO
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 decoders: List[AutoregressiveDecoder],
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 gating_strategy: str = "uniform",
                 gate_dimension: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """TODO

        Arguments:
            name: The name for the model part.
            TODO
        """
        check_argument_types()
        AutoregressiveDecoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            embedding_size=None,
            embeddings_source=None,
            tie_embeddings=False,
            label_smoothing=False,
            supress_unk=False,
            reuse=reuse,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            initializers=initializers)

        self.decoders = decoders
        self.gating_strategy = gating_strategy
        self.gate_dimension = gate_dimension

        # Check for decoder constraints
        for decoder in self.decoders:
            if decoder.vocabulary != self.vocabulary:
                raise ValueError("The expert decoders must have the same "
                                 "vocabulary")

        if self.gating_strategy not in STRATEGIES:
            raise ValueError(
                "Unknown decoder weighting strategy '{}'. "
                "Allowed: {}".format(self.gating_strategy,
                                     ", ".join(STRATEGIES)))

        if (self.gating_strategy == "context-aware"
                and self.gate_dimension is None):
            raise ValueError("Gate dimension must be defined when using "
                             "`context-aware` gating strategy")
    # pylint: enable=too-many-arguments

    @property
    def embedding_size(self) -> int:
        # TODO: concatenate embeddings from the decoders?
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def go_symbols(self) -> tf.Tensor:
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def decoding_w(self) -> tf.Variable:
        # TODO: we can concatenate output matrices and decode concatenated
        # hidden states
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def decoding_b(self) -> Optional[tf.Variable]:
        # TODO: see decoding_w
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        # TODO: concatenate embedding matrices as in decoding_w?
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def train_logits(self) -> tf.Tensor:
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def train_probs(self) -> tf.Tensor:
        probs = tf.stack([d.train_probs for d in self.decoders], -1)
        contexts = tf.concat(
            [d.train_output_states for d in self.decoders], -1)

        weights = tf.expand_dims(self.expert_weights(contexts), 2)
        return tf.reduce_sum(weights * probs, -1)

    @tensor
    def train_logprobs(self) -> tf.Tensor:
        # TODO: numerical stability of log
        return tf.log(self.train_probs)

    @tensor
    def train_xents(self) -> tf.Tensor:
        # TODO: label smoothing?
        train_targets = tf.one_hot(self.train_inputs, len(self.vocabulary))
        xent = -tf.reduce_sum(
            train_targets * self.train_logprobs, 2)
        return xent * self.train_mask

    @tensor
    def train_loss(self) -> tf.Tensor:
        return (tf.reduce_sum(self.train_xents)
                / tf.reduce_sum(self.train_mask))

    @tensor
    def runtime_logits(self) -> tf.Tensor:
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def runtime_xents(self) -> tf.Tensor:
        train_targets = tf.one_hot(self.train_inputs, len(self.vocabulary))
        min_time = tf.minimum(tf.shape(train_targets)[0],
                              tf.shape(self.runtime_logprobs)[0])

        xent = -tf.reduce_sum(
            train_targets[:min_time, :] * self.runtime_logprobs[:min_time, :],
            axis=2)
        return xent * self.train_mask[:min_time, :]

    @tensor
    def runtime_loss(self) -> tf.Tensor:
        return (tf.reduce_sum(self.runtime_xents)
                / tf.reduce_sum(self.train_mask))

    @tensor
    def runtime_probs(self) -> tf.Tensor:
        runtime_result = LoopState(*self.runtime_loop_result)
        return runtime_result.histories.other.probs

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        # TODO: numerical stability of log
        return tf.log(self.runtime_probs)

    @tensor
    def output_dimension(self) -> int:
        raise NotImplementedError("Not supported by the ExpertDecoder")

    def expert_weights(self, contexts) -> tf.Tensor:
        """Compute the weights of the decoders.

        Arguments:
            contexts: Tensors of a shape
                [length, batch_size, num_of_decoders * hidden_state_size]
                containing contexts for computing the context-aware weights.
                It is also used to infer the shape of the output tensor.

        Returns:
            Tensor of shape[length, batch_size, num_of_decoders]
        """
        if self.gating_strategy == "uniform":
            weights = tf.constant(
                [1. for _ in self.decoders], name="expert_weights")
            weights = tf.tile([[weights]], get_shape_list(contexts)[:2] + [1])
        elif self.gating_strategy == "global":
            weights = get_variable(
                name="expert_weights",
                shape=[len(self.decoders)],
                dtype=tf.float32,
                initializer=tf.ones_initializer())
            weights = tf.tile([[weights]], get_shape_list(contexts)[:2] + [1])
        elif self.gating_strategy == "context-aware":
            hidden_layer = tf.layers.dense(
                contexts, self.gate_dimension, use_bias=True,
                name="gate_hidden_layer")
            weights = tf.layers.dense(
                tf.tanh(hidden_layer), len(self.decoders), use_bias=True,
                name="expert_weights")
        else:
            raise NotImplementedError(
                "Unknown decoder weighing combination strategy: {}"
                .format(self.gating_strategy))

        return tf.nn.softmax(weights, -1)

    def get_initial_feedables(self) -> DecoderFeedables:
        expert_feedables = ExpertFeedables(
            dec_loop_states=[d.get_initial_loop_state()
                             for d in self.decoders])

        # We assign a dummy variable to the attributes not used
        # by the ExpertDecoder so the tf.while_loop can infer its shape
        dummy_var = tf.constant(0, name="feed_dummy")
        return DecoderFeedables(
            step=tf.constant(0, tf.int32),
            finished=tf.zeros([self.batch_size], dtype=tf.bool),
            embedded_input=dummy_var,
            other=expert_feedables)

    def get_initial_histories(self) -> DecoderHistories:
        output_symbols = tf.zeros(
            shape=[0, self.batch_size],
            dtype=tf.int64,
            name="hist_output_symbols")

        output_mask = tf.zeros(
            shape=[0, self.batch_size],
            dtype=tf.bool,
            name="hist_output_mask")

        expert_histories = ExpertHistories(
            probs=tf.zeros(
                shape=[0, self.batch_size, len(self.vocabulary)],
                dtype=tf.float32,
                name="hist_expert_probs"),
            weights=tf.zeros(
                shape=[0, self.batch_size, len(self.decoders)],
                dtype=tf.float32,
                name="hist_expert_weights"))


        # We assign a dummy variable to the attributes not used
        # by the ExpertDecoder so the tf.while_loop can infer its shape
        dummy_var = tf.constant(0, name="hist_dummy")
        return DecoderHistories(
            logits=dummy_var,
            output_states=dummy_var,
            output_symbols=output_symbols,
            output_mask=output_mask,
            other=expert_histories)

    def get_body(self, train_mode: bool, sample: bool = False,
                 temperature: float = 1.) -> Callable:
        decoder_bodies = [d.get_body(train_mode=train_mode)
                          for d in self.decoders]

        def is_finished(finished: tf.Tensor, symbols: tf.Tensor) -> tf.Tensor:
            has_just_finished = tf.equal(symbols, END_TOKEN_INDEX)
            return tf.logical_or(finished, has_just_finished)

        def probs_to_symbols(probs: tf.Tensor,
                             loop_state: LoopState) -> tf.Tensor:
            if sample:
                # TODO: multinomial sampling from probs
                raise ValueError("Not supported by the ExpertDecoder")
            else:
                next_symbols = tf.argmax(probs, axis=1)

            int_unfinished_mask = tf.to_int64(
                tf.logical_not(loop_state.feedables.finished))

            # Note this works only when PAD_TOKEN_INDEX is 0. Otherwise
            # this have to be rewritten
            assert PAD_TOKEN_INDEX == 0
            next_symbols = next_symbols * int_unfinished_mask

            return next_symbols

        # pylint: disable=too-many-locals
        def body(*args) -> LoopState:
            loop_state = LoopState(*args)
            histories = loop_state.histories
            feedables = loop_state.feedables

            decoder_states = feedables.other.dec_loop_states

            next_dec_states = [
                body(*state) for body, state in zip(decoder_bodies,
                                                    decoder_states)]
            next_dec_feedables = [ls.feedables for ls in next_dec_states]
            next_dec_histories = [ls.histories for ls in next_dec_states]

            # shape(time, batch, vocab)
            probs = tf.stack([tf.nn.softmax(
                                tf.expand_dims(h.logits[-1, :, :], 0),
                                axis=-1)
                              for h in next_dec_histories], -1)
            contexts = tf.concat([tf.expand_dims(h.output_states[-1, :, :], 0)
                                  for h in next_dec_histories], -1)

            weights = tf.expand_dims(self.expert_weights(contexts), 2)
            # shape(batch, vocab)
            next_probs = tf.squeeze(tf.reduce_sum(weights * probs, -1), 0)

            next_symbols = probs_to_symbols(next_probs, loop_state)
            finished = is_finished(feedables.finished, next_symbols)

            # update the loop states of the decoders
            for i, _ in enumerate(next_dec_states):
                next_dec_feedables[i] = next_dec_feedables[i]._replace(
                    finished=finished,
                    embedded_input=self.decoders[i].embed_input_symbols(
                        next_symbols))

                next_dec_states[i] = next_dec_states[i]._replace(
                    feedables=next_dec_feedables[i])

            expert_feedables = ExpertFeedables(
                dec_loop_states=next_dec_states)

            expert_histories = ExpertHistories(
                probs=append_tensor(histories.other.probs, next_probs),
                weights=append_tensor(histories.other.weights,
                                      tf.squeeze(weights, [0, 2])))

            next_feedables = feedables._replace(
                step=feedables.step + 1,
                finished=finished,
                other=expert_feedables)

            next_histories = histories._replace(
                output_symbols=append_tensor(
                    histories.output_symbols, next_symbols),
                output_mask=append_tensor(
                    histories.output_mask, tf.logical_not(finished)),
                other=expert_histories)

            return LoopState(
                feedables=next_feedables,
                histories=next_histories,
                constants=loop_state.constants)
        # pylint: enable=too-many-locals

        return body
