"""TODO."""
from typing import Callable, List, Union

import tensorflow as tf

from neuralmonkey.attention.base_attention import (
    Attendable, get_attention_states, get_attention_mask)
from neuralmonkey.attention.transformer_cross_layer import (
    serial, parallel, flat, hierarchical)
from neuralmonkey.logging import warn
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.nn.utils import dropout

STRATEGIES = ["serial", "parallel", "flat", "hierarchical"]


# We inherit from ModelPart to access self.train_mode potentially creating
# a diamond inheritance pattern in the derived class. However, this should
# be fine since we do not override any of the class methods/attributes.
# pylint: disable=too-few-public-methods
class Attentive(ModelPart):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Attendable],
                 n_heads_enc: Union[List[int], int],
                 n_heads_hier: int = None,
                 attention_combination_strategy: str = "serial",
                 dropout_keep_prob: float = 1.0,
                 attention_dropout_keep_prob: Union[float, List[float]] = 1.0,
                 use_att_transform_bias: bool = False,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Initialize the common parameters.

        Provides methods and attributes necessary for computing attention
        across the input encoders.

        Arguments:
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects.
            encoders: Input encoders for the decoder to attend to.
            n_heads_enc: Number of the attention heads over each encoder.
                Either a list which size must be equal to ``encoders``, or a
                single integer. In the latter case, the number of heads is
                equal for all encoders.
            n_heads_hier: Number of the attention heads for the second
                attention in the ``hierarchical`` attention combination.
            attention_comnbination_strategy: One of ``serial``, ``parallel``,
                ``flat``, ``hierarchical``. Controls the attention combination
                strategy for enc-dec attention.
            dropout_keep_prob: Probability of keeping a value during dropout.
            attention_dropout_keep_prob: Probability of keeping a value
                during dropout on the attention output.
            use_att_transform_bias: Add bias to the feed-forward layers in
                the attention.

        TODO:
            Generalize the attention.
        """
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoders = encoders
        self.n_heads_hier = n_heads_hier
        self.attention_combination_strategy = attention_combination_strategy
        self.dropout_keep_prob = dropout_keep_prob
        self.use_att_transform_bias = use_att_transform_bias

        if isinstance(n_heads_enc, int):
            if attention_combination_strategy == "flat":
                self.n_heads_enc = [n_heads_enc]
            else:
                self.n_heads_enc = [n_heads_enc for _ in self.encoders]
        else:
            self.n_heads_enc = n_heads_enc

        if isinstance(attention_dropout_keep_prob, float):
            self.attention_dropout_keep_prob = [
                attention_dropout_keep_prob for _ in encoders]
        else:
            self.attention_dropout_keep_prob = attention_dropout_keep_prob

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

        if any((val < 0.0 or val > 1.0)
               for val in self.attention_dropout_keep_prob):
            raise ValueError(
                "Attention dropout keep probabilities must be "
                "a real number in the interval [0,1].")
    # pylint: enable=too-many-arguments

    def encoder_attention(self, queries: tf.Tensor) -> tf.Tensor:
        """Compute attention context vectors over encoders using queries."""
        enc_states = self.encoder_states()
        enc_masks = self.encoder_masks()
        assert enc_states is not None
        assert enc_masks is not None

        # Attention dropout callbacks are created in a loop so we need to
        # use a factory function to prevent late binding.
        def make_dropout_callback(
                prob: float) -> Callable[[tf.Tensor], tf.Tensor]:
            def callback(x: tf.Tensor) -> tf.Tensor:
                return dropout(x, prob, self.train_mode)
            return callback

        dropout_cb = make_dropout_callback(self.dropout_keep_prob)
        attn_dropout_cbs = [make_dropout_callback(prob)
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

        # TODO: remove this - this is already checked in the constructor
        raise NotImplementedError(
            "Unknown attention combination strategy: {}"
            .format(self.attention_combination_strategy))
