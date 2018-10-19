from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import re

import tensorflow as tf

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.runners.base_runner import (
    Executable, ExecutionResult, NextExecute)

# pylint: disable=invalid-name
Gradients = List[Tuple[tf.Tensor, tf.Variable]]
ObjectiveWeight = Union[tf.Tensor, float, None]
# pylint: enable=invalid-name

BIAS_REGEX = re.compile(r"[Bb]ias")


class Objective(NamedTuple(
        "Objective",
        [("name", str),
         ("decoder", ModelPart),
         ("loss", tf.Tensor),
         ("gradients", Optional[Gradients]),
         ("weight", ObjectiveWeight)])):
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


# pylint: disable=too-few-public-methods,too-many-locals,too-many-arguments
class GenericTrainer:

    def __init__(self,
                 objectives: List[Objective],
                 l1_weight: float = 0.0,
                 l2_weight: float = 0.0,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:

        if var_collection is None:
            var_collection = tf.GraphKeys.TRAINABLE_VARIABLES

        if var_scopes is None:
            var_lists = [tf.get_collection(var_collection)]
        else:
            var_lists = [tf.get_collection(var_collection, scope)
                         for scope in var_scopes]

        # Flatten the list of lists
        self.var_list = [var for var_list in var_lists for var in var_list]

        with tf.variable_scope("trainer", reuse=tf.AUTO_REUSE):
            step = tf.train.get_or_create_global_step()

            if optimizer:
                self.optimizer = optimizer
            else:
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=1e-4,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                    use_locking=False)
            # pylint: disable=protected-access
            if isinstance(self.optimizer._lr, tf.Tensor):
                tf.summary.scalar("learning_rate", self.optimizer._lr,
                                  collections=["summary_train"])
            # pylint: enable=protected-access

            with tf.name_scope("regularization"):
                regularizable = [v for v in tf.trainable_variables()
                                 if not BIAS_REGEX.findall(v.name)
                                 and not v.name.startswith("vgg")
                                 and not v.name.startswith("Inception")
                                 and not v.name.startswith("resnet")]
                l1_value = sum(tf.reduce_sum(abs(v)) for v in regularizable)
                l1_cost = l1_weight * l1_value if l1_weight > 0 else 0.0

                l2_value = sum(tf.reduce_sum(v ** 2) for v in regularizable)
                l2_cost = l2_weight * l2_value if l2_weight > 0 else 0.0

            # unweighted losses for fetching
            self.losses = [o.loss for o in objectives] + [l1_value, l2_value]
            tf.summary.scalar("train_l1", l1_value,
                              collections=["summary_train"])
            tf.summary.scalar("train_l2", l2_value,
                              collections=["summary_train"])

            # log all objectives
            for obj in objectives:
                tf.summary.scalar(
                    obj.name, obj.loss, collections=["summary_train"])

            # if the objective does not have its own gradients,
            # just use TF to do the derivative
            with tf.name_scope("gradient_collection"):
                differentiable_loss_sum = sum(
                    (o.weight if o.weight is not None else 1) * o.loss
                    for o in objectives
                    if o.gradients is None) + l1_cost + l2_cost
                implicit_gradients = self._get_gradients(
                    differentiable_loss_sum)

                # objectives that have their gradients explictly computed
                other_gradients = [
                    _scale_gradients(o.gradients, o.weight)
                    for o in objectives if o.gradients is not None]

                if other_gradients:
                    gradients = _sum_gradients(
                        [implicit_gradients] + other_gradients)
                else:
                    gradients = implicit_gradients

                tf.summary.scalar("train_opt_cost",
                                  differentiable_loss_sum,
                                  collections=["summary_train"])

            if clip_norm:
                assert clip_norm > 0.0
                gradients = [(tf.clip_by_norm(grad, clip_norm), var)
                             for grad, var in gradients
                             if grad is not None]

            self.all_coders = set.union(*(obj.decoder.get_dependencies()
                                          for obj in objectives))

            self.train_op = self.optimizer.apply_gradients(
                gradients, global_step=step)

            for grad, var in gradients:
                if grad is not None:
                    tf.summary.histogram(
                        "gr_{}".format(var.name),
                        grad, collections=["summary_gradients"])

            self.histogram_summaries = tf.summary.merge(
                tf.get_collection("summary_gradients"))
            self.scalar_summaries = tf.summary.merge(
                tf.get_collection("summary_train"))

    def _get_gradients(self, tensor: tf.Tensor) -> Gradients:
        gradient_list = self.optimizer.compute_gradients(tensor, self.var_list)
        return gradient_list

    def get_executable(self,
                       compute_losses: bool = True,
                       summaries: bool = True,
                       num_sessions: int = 1) -> Executable:
        assert compute_losses
        if num_sessions != 1:
            raise ValueError(
                "Trainer only supports execution in a single session")

        return TrainExecutable(self, summaries)

def _sum_gradients(gradients_list: List[Gradients]) -> Gradients:
    summed_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for gradients in gradients_list:
        for tensor, var in gradients:
            if tensor is not None:
                if var not in summed_dict:
                    summed_dict[var] = tensor
                else:
                    summed_dict[var] += tensor
    return [(tensor, var) for var, tensor in summed_dict.items()]


def _scale_gradients(gradients: Gradients,
                     weight: ObjectiveWeight) -> Gradients:

    result = []  # type: Gradients
    for tensor, var in gradients:
        if weight is not None and tensor is not None:
            result.append((weight * tensor, var))
        else:
            result.append((tensor, var))

    return result


class TrainExecutable(Executable):

    def __init__(self, trainer: GenericTrainer, summaries: bool) -> None:
        self.trainer = trainer
        self.summaries = summaries
        self.result = None

    def next_to_execute(self) -> NextExecute:
        fetches = {"train_op": self.trainer.train_op}
        if self.summaries:
            fetches["scalar_summaries"] = self.trainer.scalar_summaries
            fetches["histogram_summaries"] = self.trainer.histogram_summaries
        fetches["losses"] = self.trainer.losses
        fetches["_update_ops"] = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return self.trainer.all_coders, fetches, [{}]

    def collect_results(self, results: List[Dict]) -> None:
        assert len(results) == 1
        result = results[0]

        scalar_summaries = (
            result["scalar_summaries"] if self.summaries else None)
        histogram_summaries = (
            result["histogram_summaries"] if self.summaries else None)

        self.result = ExecutionResult(
            [], losses=result["losses"],
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)
