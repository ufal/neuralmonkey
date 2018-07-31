from typing import Any, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import warn
from neuralmonkey.trainers.generic_trainer import (
    GenericTrainer, Objective, ObjectiveWeight)
from neuralmonkey.trainers.regularizers import (
    Regularizer, L1Regularizer, L2Regularizer)


def xent_objective(decoder, weight=None) -> Objective:
    """Get XENT objective from decoder with cost."""
    return Objective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        loss=decoder.cost,
        gradients=None,
        weight=weight,
    )

# pylint: disable=too-few-public-methods,too-many-arguments


class CrossEntropyTrainer(GenericTrainer):

    def __init__(self,
                 decoders: List[Any],
                 decoder_weights: List[ObjectiveWeight] = None,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 regularizers: List[Regularizer] = None,
                 l1_weight: float = 0.,
                 l2_weight: float = 0.,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()

        if decoder_weights is None:
            decoder_weights = [None for _ in decoders]

        if regularizers is None:
            regularizers = []
        if l1_weight > 0.:
            if L1Regularizer in [type(r) for r in regularizers]:
                warn("You specified both trainer l1_weight "
                     "and a L1Regularizer object in your config")
            regularizers.append(L1Regularizer(weight=l1_weight))

        if l2_weight > 0.:
            if L2Regularizer in [type(r) for r in regularizers]:
                warn("You specified both trainer l2_weight "
                     "and a L2Regularizer object in your config")
            regularizers.append(L2Regularizer(weight=l2_weight))

        if len(decoder_weights) != len(decoders):
            raise ValueError(
                "decoder_weights (length {}) do not match decoders (length {})"
                .format(len(decoder_weights), len(decoders)))

        objectives = [xent_objective(dec, w)
                      for dec, w in zip(decoders, decoder_weights)]

        GenericTrainer.__init__(
            self,
            objectives=objectives,
            clip_norm=clip_norm,
            optimizer=optimizer,
            regularizers=regularizers,
            var_scopes=var_scopes,
            var_collection=var_collection)
