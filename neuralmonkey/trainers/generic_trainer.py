from typing import Dict, List, Optional, Sequence
import re

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.logging import warn
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.runners.base_runner import GraphExecutor, NextExecute
from neuralmonkey.trainers.objective import (
    Objective, Gradients, ObjectiveWeight)
from neuralmonkey.trainers.regularizers import (
    Regularizer, L1Regularizer, L2Regularizer)

BIAS_REGEX = re.compile(r"[Bb]ias")


# pylint: disable=too-few-public-methods,too-many-locals,too-many-arguments
class GenericTrainer(GraphExecutor, Feedable):

    class Executable(GraphExecutor.Executable["GenericTrainer"]):

        def __init__(self, executor: "GenericTrainer", compute_losses: bool,
                     summaries: bool, num_sessions: int) -> None:
            assert compute_losses
            if num_sessions != 1:
                raise ValueError(
                    "Trainer only supports execution in a single session")

            super().__init__(executor, compute_losses, summaries, num_sessions)

        def next_to_execute(self) -> NextExecute:
            fetches = self.executor.fetches

            if self.summaries:
                fetches.update(self.executor.summaries)

            return fetches, []

        def collect_results(self, results: List[Dict]) -> None:
            assert len(results) == 1
            result = results[0]

            summaries = []
            if self.summaries:
                summaries.extend([result["scalar_summaries"],
                                  result["histogram_summaries"]])

            objective_names = [obj.name for obj in self.executor.objectives]
            objective_names += ["L1", "L2"]

            losses = dict(zip(objective_names, result["losses"]))

            self.set_result({}, losses, result["batch_size"], summaries)

    @staticmethod
    def default_optimizer():
        return tf.train.AdamOptimizer(learning_rate=1e-4)

    def __init__(self,
                 objectives: Sequence[Objective],
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 regularizers: List[Regularizer] = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()
        GraphExecutor.__init__(self, {obj.decoder for obj in objectives})
        Feedable.__init__(self)

        self.objectives = objectives
        self.clip_norm = clip_norm
        self.var_scopes = var_scopes
        self.var_collection = var_collection
        if self.var_collection is None:
            self.var_collection = tf.GraphKeys.TRAINABLE_VARIABLES

        self.objectives = objectives

        self.regularizers = []  # type: List[Regularizer]
        if regularizers is not None:
            self.regularizers = regularizers

        self.optimizer = (
            optimizer if optimizer is not None else self.default_optimizer())

    # pylint: disable=no-self-use
    @tensor
    def regularization_losses(self) -> List[tf.Tensor]:
        """Compute the regularization losses, e.g. L1 and L2."""
        regularizable = [v for v in tf.trainable_variables()
                         if not BIAS_REGEX.findall(v.name)
                         and not v.name.startswith("vgg")
                         and not v.name.startswith("Inception")
                         and not v.name.startswith("resnet")]

        if not regularizable:
            warn("It seems that there are no trainable variables in the model")
            return tf.zeros([]), tf.zeros([])

        with tf.name_scope("regularization"):
            reg_values = [reg.value(regularizable)
                          for reg in self.regularizers]

        return reg_values
    # pylint: enable=no-self-use

    @tensor
    def objective_values(self) -> List[tf.Tensor]:
        """Compute unweighted losses for fetching."""
        # Note here we need to call the losses first, in case the model is
        # being built. We need to compute the regularizers after that.
        losses = [o.loss for o in self.objectives]

        return losses + self.regularization_losses

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

        obj_weights += [reg.weights for reg in self.regularizers]
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

        reg_values = self.regularization_losses
        # we always want to include l2 values in the summary
        if L1Regularizer not in [type(r) for r in self.regularizers]:
            l1_reg = L1Regularizer(name="train_l1", weight=0.)
            tf.summary.scalar(l1_reg.name, l1_reg.value(regularizable),
                              collections=["summary_train"])
        if L2Regularizer not in [type(r) for r in self.regularizers]:
            l2_reg = L2Regularizer(name="train_l2", weight=0.)
            tf.summary.scalar(l2_reg.name, l2_reg.value(regularizable),
                              collections=["summary_train"])

        for reg, reg_value in zip(self.regularizers, reg_values):
            tf.summary.scalar(reg.name, reg_value,
                              collections=["summary_train"])

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

    @property
    def fetches(self) -> Dict[str, tf.Tensor]:
        return {"train_op": self.train_op,
                "losses": self.objective_values,
                "batch_size": self.batch_size,
                "_update_ops": tf.get_collection(tf.GraphKeys.UPDATE_OPS)}
