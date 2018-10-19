from typing import Dict, List, Tuple
# pylint: disable=unused-import
from typing import Optional
# pylint: enable=unused-import

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decorators import tensor
from neuralmonkey.runners.base_runner import (
    Executable, ExecutionResult, NextExecute)
from neuralmonkey.trainers.generic_trainer import (
    GenericTrainer, Objective, Gradients)


class DelayedUpdateTrainer(GenericTrainer):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 batches_per_update: int,
                 objectives: List[Objective],
                 l1_weight: float = 0.0,
                 l2_weight: float = 0.0,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()
        GenericTrainer.__init__(self, objectives, l1_weight, l2_weight,
                                clip_norm, optimizer, var_scopes,
                                var_collection)

        self.batches_per_update = batches_per_update
    # pylint: enable=too-many-arguments

    @tensor
    def existing_grads_and_vars(self) -> Tuple[
            List[tf.Tensor], List[tf.Variable]]:
        orig_grads = super().raw_gradients

        # pylint: disable=not-an-iterable
        # Pylint does not understand @tensor annotations
        transposed = tuple(zip(
            *[(grad, var) for grad, var in orig_grads if grad is not None]))
        # pylint: enable=not-an-iterable

        return list(transposed[0]), list(transposed[1])

    @tensor
    def gradient_buffers(self) -> List[tf.Variable]:
        # pylint: disable=unpacking-non-sequence
        existing_gradients, _ = self.existing_grads_and_vars
        # pylint: enable=unpacking-non-sequence

        with tf.variable_scope("gradient_buffer"):
            return [tf.Variable(initial_value=tf.zeros_like(grad),
                                trainable=False)
                    for grad in existing_gradients]

    @tensor
    def objective_buffers(self) -> List[tf.Variable]:
        with tf.variable_scope("loss_buffers"):
            return [tf.Variable(0.0, trainable=False) for _ in self.objectives]

    # pylint: disable=no-self-use
    @tensor
    def diff_buffer(self) -> tf.Variable:
        return tf.Variable(0.0, trainable=False)

    @tensor
    def cumulator_counter(self) -> tf.Variable:
        return tf.Variable(0, trainable=False, name="self.cumulator_counter")
    # pylint: enable=no-self-use

    @tensor
    def accumulate_ops(self) -> List[tf.Operation]:
        # pylint: disable=unpacking-non-sequence
        existing_gradients, _ = self.existing_grads_and_vars
        # pylint: enable=unpacking-non-sequence

        # pylint: disable=not-an-iterable
        # Pylint does not understand @tensor annotations
        accumulate_ops = [
            tf.assign_add(gradbuf, grad)
            for gradbuf, grad in zip(
                self.gradient_buffers, existing_gradients)]

        accumulate_ops.extend(
            tf.assign_add(objbuf, obj.loss)
            for objbuf, obj in zip(self.objective_buffers, self.objectives))
        # pylint: enable=not-an-iterable

        accumulate_ops.append(
            tf.assign_add(self.diff_buffer, self.differentiable_loss_sum))
        accumulate_ops.append(
            tf.assign_add(self.cumulator_counter, 1))

        return accumulate_ops

    @tensor
    def reset_ops(self) -> List[tf.Operation]:
        # pylint: disable=not-an-iterable
        # Pylint does not understand @tensor annotations
        reset_ops = [tf.assign(gradbuf, tf.zeros_like(gradbuf))
                     for gradbuf in self.gradient_buffers]
        reset_ops.extend(
            tf.assign(objbuf, 0.0) for objbuf in self.objective_buffers)
        # pylint: enable=not-an-iterable

        reset_ops.append(tf.assign(self.diff_buffer, 0.0))
        reset_ops.append(tf.assign(self.cumulator_counter, 0))
        return reset_ops

    @tensor
    def raw_gradients(self) -> Gradients:
        """Return averaged gradients over buffers"""
        # pylint: disable=not-an-iterable
        # Pylint does not understand @tensor annotations
        averaged_grads = [grad / tf.to_float(self.cumulator_counter)
                          for grad in self.gradient_buffers]
        # pylint: enable=not-an-iterable

        tf.summary.scalar(
            "train_opt_cost",
            self.diff_buffer / tf.to_float(self.cumulator_counter),
            collections=["summary_train"])

        # log all objectives
        for obj, objbuf in zip(self.objectives, self.objective_buffers):
            tf.summary.scalar(
                obj.name, objbuf / tf.to_float(self.cumulator_counter),
                collections=["summary_train"])

        # now, zip averaged grads with associated vars to a Gradients struct.
        # pylint: disable=unpacking-non-sequence
        _, existing_vars = self.existing_grads_and_vars
        # pylint: enable=unpacking-non-sequence
        return list(zip(averaged_grads, existing_vars))

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

        return DelayedTrainExecutable(self, summaries)


class DelayedTrainExecutable(Executable):

    def __init__(self, trainer: DelayedUpdateTrainer, summaries: bool) -> None:
        self.trainer = trainer
        self.summaries = summaries
        self.result = None  # type: Optional[ExecutionResult]

        self.state = 0
        self.res_hist_sums = None
        self.res_scal_sums = None
        self.res_losses = None

    def next_to_execute(self) -> NextExecute:

        if self.state == 0:  # ACCUMULATING
            fetches = {"accumulators": self.trainer.accumulate_ops,
                       "counter": self.trainer.cumulator_counter,
                       "losses": self.trainer.objective_values}
            coders = self.trainer.all_coders

        elif self.state == 1:  # UPDATING
            fetches = {
                "train_op": self.trainer.train_op,
                "_update_ops": tf.get_collection(tf.GraphKeys.UPDATE_OPS)}

            if self.summaries:
                fetches.update(self.trainer.summaries)

            coders = set()

        else:  # RESETTING
            fetches = {"resets": self.trainer.reset_ops}
            coders = set()

        return coders, fetches, [{}]

    def collect_results(self, results: List[Dict]) -> None:
        assert len(results) == 1
        result = results[0]

        if self.state == 0:  # ACCUMULATING
            self.res_losses = result["losses"]

            # Are we updating?
            counter = result["counter"]

            if counter == self.trainer.batches_per_update:
                self.state = 1
                return
        elif self.state == 1:
            if self.summaries:
                self.res_scal_sums = result["scalar_summaries"]
                self.res_hist_sums = result["histogram_summaries"]

            self.state = 2
            return

        assert self.res_losses is not None
        self.result = ExecutionResult(
            [], losses=self.res_losses,
            scalar_summaries=self.res_scal_sums,
            histogram_summaries=self.res_hist_sums,
            image_summaries=None)
