import tensorflow as tf
from tensorflow.python.framework import ops
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.experiment import Experiment


def reverse_gradient(x: tf.Tensor) -> tf.Tensor:
    """Flips the sign of the incoming gradient during training."""

    grad_name = "gradient_reversal_{}".format(x.name)

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) ]

    graph = Experiment.get_current().graph
    with graph.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(x)

    return y


def reverse_gradients_objective(objective: Objective) -> Objective:
    """Get an objective with the same value, but reversed gradients."""
    check_argument_types()
    if objective.gradients is not None:
        raise ValueError(
            "Objective for gradient_reversal_cannot have explicit gradients.")

    return Objective(
        name=objective.name + "_reversed",
        decoder=objective.decoder,
        loss=reverse_gradient(objective.loss),
        gradients=None,
        weight=objective.weight)
