from typing import Any, List

from neuralmonkey.trainers.generic_trainer import GenericTrainer, Objective

# tests: lint, mypy


def xent_objective(decoder) -> Objective:
    """Get XENT objective from decoder with cost."""
    return Objective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        loss=decoder.cost,
        gradients=None
    )

# pylint: disable=too-few-public-methods


class CrossEntropyTrainer(GenericTrainer):

    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None) -> None:
        objectives = [xent_objective(dec) for dec in decoders]
        super(CrossEntropyTrainer, self).__init__(
            objectives, l1_weight, l2_weight, clip_norm=clip_norm,
            optimizer=optimizer)
