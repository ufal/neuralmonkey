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
    """The training objective base class."""

    def __init__(self, name: str, decoder: MP) -> None:
        """Construct the objective.

        Arguments:
            name: The name for the objective. This will be used e.g. in
                TensorBoard.
        """
        self._name = name
        self._decoder = decoder

    @property
    def decoder(self) -> MP:
        """Get the decoder used by the objective."""
        return self._decoder

    @property
    def name(self) -> str:
        """Get the name of the objective."""
        return self._name

    @abstractproperty
    def loss(self) -> tf.Tensor:
        """Return the loss tensor fetched by the trainer."""
        raise NotImplementedError()

    @property
    def gradients(self) -> Optional[Gradients]:
        """Manually specified gradients - useful for reinforcement learning."""
        return None

    @property
    def weight(self) -> Optional[tf.Tensor]:
        """Return the weight of this objective.

        The loss will be multiplied by this so the gradients can be controlled
        in case of multiple objectives.

        Returns:
            An optional tensor. If None, default weight of 1 is assumed.
        """
        return None


class CostObjective(Objective[GenericModelPart]):
    """Cost objective class.

    This class represent objectives that are based directly on a `cost`
    attribute of any compatible model part.
    """

    def __init__(self, decoder: GenericModelPart,
                 weight: ObjectiveWeight = None) -> None:
        """Construct a new instance of the `CostObjective` class.

        Arguments:
            decoder: A `GenericModelPart` instance that has a `cost` attribute.
            weight: The weight of the objective.

        Raises:
            `TypeError` when the decoder argument does not have the `cost`
            attribute.
        """
        check_argument_types()
        if "cost" not in dir(decoder):
            raise TypeError("The decoder does not have the 'cost' attribute")

        name = "{} - cost".format(str(decoder))

        super().__init__(name, decoder)
        self._weight = weight

    @tensor
    def loss(self) -> tf.Tensor:
        return getattr(self.decoder, "cost")

    @tensor
    def weight(self) -> Optional[tf.Tensor]:
        if self._weight is None:
            return None

        if isinstance(self._weight, float):
            return tf.constant(self._weight)

        return self._weight
