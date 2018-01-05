from typing import Any, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import (
    GenericTrainer, Objective, ObjectiveWeight)
from neuralmonkey.trainers.optimizers import OptimizerGetter, adam_optimizer
from neuralmonkey.trainers.lr_decay import DecayFunction


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
                 l1_weight: float = 0.,
                 l2_weight: float = 0.,
                 clip_norm: float = None,
                 optimizer_getter: OptimizerGetter = adam_optimizer(),
                 global_step: tf.Tensor = None,
                 decay_function: DecayFunction = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()

        if decoder_weights is None:
            decoder_weights = [None for _ in decoders]

        if len(decoder_weights) != len(decoders):
            raise ValueError(
                "decoder_weights (length {}) do not match decoders (length {})"
                .format(len(decoder_weights), len(decoders)))

        objectives = [xent_objective(dec, w)
                      for dec, w in zip(decoders, decoder_weights)]

        GenericTrainer.__init__(
            self,
            objectives=objectives,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            clip_norm=clip_norm,
            optimizer_getter=optimizer_getter,
            global_step=global_step,
            decay_function=decay_function,
            var_scopes=var_scopes,
            var_collection=var_collection)
