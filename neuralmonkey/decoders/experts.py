import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.autoregressive import (
    AutoregressiveDecoder, LoopState)
from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.tf_utils import (
    append_tensor, gather_flat, get_state_shape_invariants, partial_transpose,
    get_shape_list)
from neuralmonkey.vocabulary import (
    Vocabulary, END_TOKEN_INDEX, PAD_TOKEN_INDEX)


STRATEGIES = ["uniform", "global", "context-aware"]


class ExpertFeedables(NamedTuple(
        "ExpertFeedables", [
            ("step", tf.Tensor),
            ("finished", tf.Tensor),
            ("input_symbol", tf.Tensor),
            ("prev_logits", tf.Tensor),
            ("dec_loop_states", List[LoopState])]))


class ExpertDecoder(AutoregressiveDecoder):
    """In-graph mixture of experts decocer.

    TODO
    """

    def __init__(self,
                 name: str,
                 decoders: List[AutoregressiveDecoder],
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_output_len: int,
                 gating_strategy: str = "uniform",
                 gate_dimension: int = None,
                 dropout_keep_prob: float = 1.0,
                 reuse: ModelPart = None
                 save_checkpoint: str = None
                 load_checkpoint: str = None
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
            embeddings_source=None
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
    def get_logits(self, state: tf.Tensor) -> tf.Tensor:
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
    def train_xents(self) -> tf.Tensor
        # TODO: label smoothing?
        train_targets = tf.one_hot(self.train_inputs, len(self.vocabulary))
        xent = -tf.reduce_sum(
            train_targets * self.train_logprobs, 2)
        masked_xent = xent * self.train_mask
        return tf.reduce_mean(masked_xent, 0)

    @tensor
    def runtime_logits(self) -> tf.Tensor:
        raise NotImplementedError("Not supported by the ExpertDecoder")

    @tensor
    def runtime_xents(self) -> tf.Tensor:
        train_targets = tf.one_hot(self.train_inputs, len(self.vocabulary))
        min_time = tf.minimum(tf.shape(train_targets)[0],
                              tf.shape(self.runtime_logprobs)[0])

        xent = -tf.reduce_sum(
            train_targets[:min_time, :] * self.runtime_logprobs[:min_time, :], 2)
        masked_xent = xent * self.train_mask[:min_time, :]
        return tf.reduce_mean(masked_xent, 0)

    @tensor
    def runtime_probs(self) -> tf.Tensor:
        #TODO

    @tensor
    def runtime_logprobs(self) -> tf.Tensor:
        # TODO: numerical stability of log
        return tf.log(self.runtime_probs)

    @tensor
    def output_dimension(self) -> int:
        # TODO

    @tensor
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
            weights = tf.tile([[weights]], contexts.get_shape()[:2] + [1])
        elif self.gating_strategy == "global":
            weights = get_variable(
                name="expert_weights"
                shape=[len(self.decoders)],
                dtype=tf.float32,
                initializer=tf.ones_initializer())
            weights = tf.tile([[weights]], contexts.get_shape()[:2] + [1])
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

        return tf.softmax(weights, -1)

    def get_initial_loop_state(self) -> LoopState:
        default_ls = AutoregressiveDecoder.get_initial_loop_state(self)
        feedables = default_ls.feedables
        histories = default_ls.histories

        feedables = feedables._replace(
            dec_loop_states=[d.get_initial_loop_state()
                             for d in self.decoders],
            prev_logits=None)

        histories = histories._replace(
            logits=None
            decoder_outputs=None)

        return LoopState(
            feedables=feedables,
            histories=histories,
            constants=default_ls.constants)

    def get_body(self, train_mode: bool, sample: bool = False,
                 temperature: float = 1.) -> Callable:
        decoder_bodies = [d.get_body(train_mode=train_mode)
                          for d in self.decoders]

        # pylint: disable=too-many-locals
        def body(*args) -> LoopState:
            loop_state = LoopState(*args)
            histories = loop_state.histories
            feedables = loop_state.feedables
            
            decoder_states = feedables.dec_loop_states
            
            next_dec_states = [
                body(*state) for body, state in zip(decoder_bodies,
                                                    decoder_states)
            next_dec_feedables = [ls.feedables for ls in next_dec_states]
            next_dec_histories = [ls.histories for ls in next_dec_states]

            probs = tf.stack(
                [tf.nn.softmax(tf.expand_dims(f.prev_logits, 0))
                 for f in next_dec_feedables], -1)
            contexts = tf.concat([h.decoder_outputs[-1, :, :]
                                  for h in next_dec_histories], -1)

            weights = tf.expand_dims(self.expert_weights(contexts), 2)
            next_probs = tf.reduce_sum(weights * probs, -1)

            if sample:
                # TODO: multinomial sampling from next_probs
                raise ValueError("Not supported by the ExpertDecoder")
            else:
                next_symbols = tf.argmax(next_probs, axis=1)

                # TODO: why do we mask only in this if-else branch?
                int_unfinished_mask = tf.to_int64(
                    tf.logical_not(feedables.finished))

                # Note this works only when PAD_TOKEN_INDEX is 0. Otherwise
                # this have to be rewritten
                assert PAD_TOKEN_INDEX == 0
                next_symbols = next_symbols * int_unfinished_mask 

            has_just_finished = tf.equal(next_symbols, END_TOKEN_INDEX)
            has_finished = tf.logical_or(feedables.finished,
                                         has_just_finished)
            not_finished = tf.logical_not(has_finished)

            # update the loop states of the decoders
            # TODO: check if the states are really updated
            for i, _ in enumerate(next_dec_states):
                next_dec_feedables[i] = next_dec_feedables[i]._replace(
                    finished=has_finished,
                    input_symbol=next_symbol)

                next_dec_states[i] = next_dec_states[i]._replace(
                    feedables=dec_feed)

            next_feedables = DecoderFeedables(
                step=feedables.step + 1,
                finished=has_finished,
                input_symbol=next_symbol,
                prev_logits=None,
                dec_loop_states=dec_loop_states)

            next_histories = DecoderHistories(
                logits=None,
                decoder_outputs=None,
                mask=append_tensor(feedables.mask, has_finished),
                outputs=append_tensor(feedables.outputs, next_symbol))
