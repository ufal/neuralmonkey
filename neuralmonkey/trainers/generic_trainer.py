from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import re

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.logging import log
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

    @staticmethod
    def default_optimizer():
        return tf.train.AdamOptimizer(learning_rate=1e-4)

    def __init__(self,
                 objectives: List[Objective],
                 l1_weight: float = 0.0,
                 l2_weight: float = 0.0,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()

        self.objectives = objectives
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.clip_norm = clip_norm
        self.var_scopes = var_scopes
        self.var_collection = var_collection
        if self.var_collection is None:
            self.var_collection = tf.GraphKeys.TRAINABLE_VARIABLES

        self.optimizer = (
            optimizer if optimizer is not None else self.default_optimizer())
        self.all_coders = set.union(*(obj.decoder.get_dependencies()
                                      for obj in objectives))

        log("Train op: {}".format(str(self.train_op)))

    # pylint: disable=no-self-use
    @tensor
    def regularization_losses(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the regularization losses, e.g. L1 and L2."""
        regularizable = [v for v in tf.trainable_variables()
                         if not BIAS_REGEX.findall(v.name)
                         and not v.name.startswith("vgg")
                         and not v.name.startswith("Inception")
                         and not v.name.startswith("resnet")]

        with tf.name_scope("regularization"):
            l1_norm = sum(tf.reduce_sum(abs(v)) for v in regularizable)
            l2_norm = sum(tf.reduce_sum(v ** 2) for v in regularizable)

        return l1_norm, l2_norm
    # pylint: enable=no-self-use

    @tensor
    def objective_values(self) -> List[tf.Tensor]:
        """Compute unweighted losses for fetching."""
        # pylint: disable=unpacking-non-sequence
        l1_norm, l2_norm = self.regularization_losses
        # pylint: disable=unpacking-non-sequence

        return [o.loss for o in self.objectives] + [l1_norm, l2_norm]

    @tensor
    def differentiable_loss_sum(self) -> tf.Tensor:
        """Compute the differentiable loss (including regularization)."""
        obj_weights = []  # type: List[Optional[float]]
        for obj in self.objectives:
            if obj.gradients is not None:
                obj_weights.append(None)
            elif obj.weight is None:
                obj_weights.append(1.0)
            else:
                obj_weights.append(obj.weight)

        obj_weights += [self.l1_weight, self.l2_weight]
        diff_loss = sum(
            o * w for o, w in zip(self.objective_values, obj_weights)
            if w is not None)

        return diff_loss

    @tensor
    def raw_gradients(self) -> Gradients:
        """Compute the gradients."""
        with tf.name_scope("gradient_collection"):
            gradients = self.optimizer.compute_gradients(
                self.differentiable_loss_sum, self.var_list)

            def scale_grads(gradients: Gradients,
                            weight: ObjectiveWeight) -> Gradients:
                result = []  # type: Gradients
                for grad, var in gradients:
                    if weight is not None and grad is not None:
                        result.append((weight * grad, var))
                    else:
                        result.append((grad, var))
                return result

            # objectives that have their gradients explictly computed
            other_gradients = [
                scale_grads(o.gradients, o.weight)
                for o in self.objectives if o.gradients is not None]

            def sum_grads(gradients_list: List[Gradients]) -> Gradients:
                summed_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
                for gradients in gradients_list:
                    for grad, var in gradients:
                        if grad is not None:
                            if var not in summed_dict:
                                summed_dict[var] = grad
                            else:
                                summed_dict[var] += grad

                return [(grad, var) for var, grad in summed_dict.items()]

            if other_gradients:
                gradients = sum_grads([gradients] + other_gradients)

        return gradients

    @tensor
    def gradients(self) -> Gradients:
        gradients = self.raw_gradients

        if self.clip_norm:
            assert self.clip_norm > 0.0
            # pylint: disable=not-an-iterable
            # Pylint does not understand @tensor annotations
            gradients = [
                (tf.clip_by_norm(grad, self.clip_norm), var)
                for grad, var in self.raw_gradients if grad is not None]
            # pylint: disable=not-an-iterable

        return gradients

    @tensor
    def train_op(self) -> tf.Operation:
        """Construct the training op."""
        with tf.name_scope("trainer"):
            step = tf.train.get_or_create_global_step()
            return self.optimizer.apply_gradients(self.gradients, step)

    @property
    def var_list(self) -> List[tf.Variable]:
        if self.var_scopes is None:
            vlists = [tf.get_collection(self.var_collection)]
        else:
            vlists = [tf.get_collection(self.var_collection, scope)
                      for scope in self.var_scopes]

        # Flatten the list of lists
        return [var for var_list in vlists for var in var_list]

    @tensor
    def summaries(self) -> Dict[str, tf.Tensor]:

        # pylint: disable=protected-access
        if isinstance(self.optimizer._lr, tf.Tensor):
            tf.summary.scalar("learning_rate", self.optimizer._lr,
                              collections=["summary_train"])
        # pylint: enable=protected-access

        # pylint: disable=unpacking-non-sequence
        l1_norm, l2_norm = self.regularization_losses
        # pylint: enable=unpacking-non-sequence
        tf.summary.scalar("train_l1", l1_norm, collections=["summary_train"])
        tf.summary.scalar("train_l2", l2_norm, collections=["summary_train"])

        for obj in self.objectives:
            tf.summary.scalar(obj.name, obj.loss,
                              collections=["summary_train"])

        tf.summary.scalar("train_opt_cost", self.differentiable_loss_sum,
                          collections=["summary_train"])

        # pylint: disable=not-an-iterable
        # Pylint does not understand @tensor annotations
        for grad, var in self.gradients:
            if grad is not None:
                summary_name = "gr_{}".format(var.name)
                tf.summary.histogram(
                    summary_name, grad, collections=["summary_gradients"])
        # pylint: enable=not-an-iterable

        return {
            "scalar_summaries": tf.summary.merge(
                tf.get_collection("summary_train")),
            "histogram_summaries": tf.summary.merge(
                tf.get_collection("summary_gradients"))}

    def get_executable(self,
                       compute_losses: bool = True,
                       summaries: bool = True,
                       num_sessions: int = 1) -> Executable:
        assert compute_losses
        if num_sessions != 1:
            raise ValueError(
                "Trainer only supports execution in a single session")

        return TrainExecutable(self, summaries)


class TrainExecutable(Executable):

    def __init__(self, trainer: GenericTrainer, summaries: bool) -> None:
        self.trainer = trainer
        self.summaries = summaries
        self.result = None  # type: Optional[ExecutionResult]

    def next_to_execute(self) -> NextExecute:
        fetches = {"train_op": self.trainer.train_op}

        if self.summaries:
            fetches.update(self.trainer.summaries)

        fetches["losses"] = self.trainer.objective_values
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
