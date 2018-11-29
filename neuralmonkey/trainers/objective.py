from abc import abstractproperty
from typing import TypeVar, Union, Tuple, List, Optional, Generic
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.model.model_part import GenericModelPart

# pylint: disable=invalid-name
ObjectiveWeight = Union[tf.Tensor, float, None]
Gradients = List[Tuple[tf.Tensor, tf.Variable]]
MP = TypeVar("MP", bound=GenericModelPart)
# pylint: enable=invalid-name


class Objective(Generic[MP]):
    """The training objective.

    Attributes:
        name: The name for the objective. Used in TensorBoard.
        decoder: The decoder which generates the value to optimize.
        loss: The loss tensor fetched by the trainer.
        gradients: Manually specified gradients. Useful for reinforcement
            learning.
        weight: The weight of this objective. The loss will be multiplied by
            this so the gradients can be controled in case of multiple
            objectives.
    """

    def __init__(self, name: str, decoder: MP) -> None:
        self._name = name
        self._decoder = decoder

    @property
    def decoder(self) -> MP:
        return self._decoder

    @property
    def name(self) -> str:
        return self._name

    @abstractproperty
    def loss(self) -> tf.Tensor:
        raise NotImplementedError()

    @property
    def gradients(self) -> Optional[Gradients]:
        return None

    @property
    def weight(self) -> Optional[tf.Tensor]:
        return None


class CostObjective(Objective[GenericModelPart]):

    def __init__(self, decoder: GenericModelPart,
                 weight: ObjectiveWeight = None) -> None:
        check_argument_types()

        name = "{} - cost".format(str(decoder))
        Objective[GenericModelPart].__init__(self, name, decoder)
        self._weight = weight

    @tensor
    def loss(self) -> tf.Tensor:
        if not hasattr(self.decoder, "cost"):
            raise TypeError("The decoder does not have the 'cost' attribute")

        return getattr(self.decoder, "cost")

    @tensor
    def weight(self) -> Optional[tf.Tensor]:
        if self._weight is None:
            return None

        if isinstance(self._weight, float):
            return tf.constant(self._weight)

        return self._weight
