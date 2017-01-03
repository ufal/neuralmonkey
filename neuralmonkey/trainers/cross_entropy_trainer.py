from typing import Any, List, Optional

from neuralmonkey.trainers.generic_trainer import (GenericTrainer, Objective,
                                                   ObjectiveWeight)

# tests: lint, mypy


def xent_objective(decoder, weight=None) -> Objective:
    """Get XENT objective from decoder with cost."""
    return Objective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        loss=decoder.cost,
        gradients=None,
        weight=weight,
    )

# pylint: disable=too-few-public-methods


class CrossEntropyTrainer(GenericTrainer):

    def __init__(self, decoders: List[Any],
                 decoder_weights: Optional[List[ObjectiveWeight]]=None,
                 l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None, global_step=None) -> None:

        if decoder_weights is None:
            decoder_weights = [None for _ in decoders]

        if len(decoder_weights) != len(decoders):
            raise ValueError(
                "decoder_weights (length {}) do not match decoders (length {})"
                .format(len(decoder_weights), len(decoders)))

        objectives = [xent_objective(dec, w)
                      for dec, w in zip(decoders, decoder_weights)]
        super(CrossEntropyTrainer, self).__init__(
            objectives, l1_weight, l2_weight, clip_norm=clip_norm,
            optimizer=optimizer, global_step=global_step)
