from typing import Dict, List

import tensorflow as tf

from neuralmonkey.runners.base_runner import (
    Executable, ExecutionResult, NextExecute)
from neuralmonkey.trainers.generic_trainer import (
    _sum_gradients, _scale_gradients, BIAS_REGEX, Objective)


class DelayedUpdateTrainer:


    @property
    def _var_list(self) -> List[tf.Variable]:

        if self.var_scopes is None:
            vlists = [tf.get_collection(self.var_collection)]
        else:
            vlists = [tf.get_collection(self.var_collection, scope)
                      for scope in self.var_scopes]

        # Flatten the list of lists
        return [var for var_list in vlists for var in var_list]

    # @property
    # def obj_weights(self) -> List[float]:
    #     return [o.weight if o.weight is not None else 1.0
    #             for o in self.objectives] + [self.l1_weight, self.l2_weight]

    # @tensor
    # def regularization_losses(self) -> Tuple[tf.Tensor, tf.Tensor]:
    #     regularizable = [v for v in tf.trainable_variables()
    #                      if not BIAS_REGEX.findall(v.name)
    #                      and not v.name.startswith("vgg")
    #                      and not v.name.startswith("Inception")
    #                      and not v.name.startswith("resnet")]

    #     with tf.name_scope("regularization"):
    #         l1 = sum(tf.reduce_sum(abs(v)) for v in regularizable)
    #         l2 = sum(tf.reduce_sum(v ** 2) for v in regularizable)

    #     return l1, l2

    # @tensor
    # def objective_values(self) -> List[tf.Tensor]:
    #     # log all objectives
    #     for o in self.objectives:
    #         tf.summary.scalar(o.name, o.loss, collections=["summary_train"])

    #     l1, l2 = self.regularization_losses

    #     tf.summary.scalar("train_l1", l1, collections=["summary_train"])
    #     tf.summary.scalar("train_l2", l2, collections=["summary_train"])

    #     # unweighted losses for fetching
    #     return [o.loss for o in self.objectives] + [l1, l2]

    # @tensor
    # def differentiable_loss_sum(self) -> tf.Tensor:

    #     with tf.name_scope("gradient_collection"):
    #         l1, l2 = self.regularization_losses
    #         regularization = l1 * self.l1_weight + l2 * self.l2_weight

    #         diff_loss = sum(
    #             (o.weight if o.weight is not None else 1.0) * o.loss
    #             for o in self.objectives if o.gradients is None)
    #         diff_loss += regularization

    #         tf.summary.scalar(
    #             "train_opt_cost", diff_loss, collections=["summary_train"])

    #     return diff_loss

    # @tensor
    # def gradients(self) -> Gradients:

    #     with tf.name_scope("gradient_collection"):
    #         gradients = self.optimizer.compute_gradients(
    #             self.differentiable_loss_sum(), self._var_list)

    #         def scale_grads(gradients: Gradients,
    #                         weight: ObjectiveWeight) -> Gradients:
    #             result = []  # type: Gradients
    #             for tensor, var in gradients:
    #                 if weight is not None and tensor is not None:
    #                     result.append((weight * tensor, var))
    #                 else:
    #                     result.append((tensor, var))
    #             return result

    #         # objectives that have their gradients explictly computed
    #         other_gradients = [
    #             scale_grads(o.gradients, o.weight)
    #             for o in self.objectives if o.gradients is not None]

    #         def sum_grads(gradients_list: List[Gradients]) -> Gradients:
    #             summed_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    #             for gradients in gradients_list:
    #                 for tensor, var in gradients:
    #                     if tensor is not None:
    #                         if var not in summed_dict:
    #                             summed_dict[var] = tensor
    #                         else:
    #                             summed_dict[var] += tensor

    #             return [(tensor, var) for var, tensor in summed_dict.items()]

    #         if other_gradients:
    #             gradients = sum_grads([gradients] + other_gradients)

    #         if self.clip_norm:
    #             gradients = [(tf.clip_by_norm(grad, self.clip_norm), var)
    #                          for grad, var in gradients if grad is not None]

    #         for grad, var in gradients:
    #             if grad is not None:
    #                 summary_name = "gr_{}".format(var.name)
    #                 tf.summary.histogram(
    #                     summary_name, grad, collections=["summary_gradients"])

    #     return gradients


    # @tensor
    # def gradient_buffers(self) -> List[tf.Variable]:

    #     existing_gradients, existing_vars = zip(*[
    #         (grad, var) for grad, var in gradients if grad is not None])

    #     return [tf.Variable(initial_value=tf.zeros_like(grad),
    #                         trainable=False)
    #             for grad in self.existing_gradients]



    # @tensor
    # def reset_ops(self) -> List[tf.Operation]:
    #     return [tf.assign(gradbuf, tf.zeros_like(gradbuf))
    #             for gradbuf in self.gradient_buffers]

    # @tensor
    # def train_op(self) -> tf.Operation:

    #     with tf.name_scope("trainer"):
    #         step = tf.train.get_or_create_global_step()

    #         # pylint: disable=protected-access
    #         if isinstance(self.optimizer._lr, tf.Tensor):
    #             tf.summary.scalar("learning_rate", self.optimizer._lr,
    #                               collections=["summary_train"])
    #         # pylint: enable=protected-access

    #         train_op = self.optimizer.apply_gradients(self.gradients, step)

    #     return train_op



    def __init__(self,
                 batches_per_update: int,
                 objectives: List[Objective],
                 l1_weight: float = 0.0,
                 l2_weight: float = 0.0,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:

        self.batches_per_update = batches_per_update
        self.objectives = objectives
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.clip_norm = clip_norm
        self.optimizer = optimizer
        self.var_scopes = var_scopes
        self.var_collection = var_collection

        if var_collection is None:
            var_collection = tf.GraphKeys.TRAINABLE_VARIABLES

        if var_scopes is None:
            var_lists = [tf.get_collection(var_collection)]
        else:
            var_lists = [tf.get_collection(var_collection, scope)
                         for scope in var_scopes]

        # Flatten the list of lists
        self.var_list = [var for var_list in var_lists for var in var_list]


        self.all_coders = set.union(*(obj.decoder.get_dependencies()
                                      for obj in objectives))

        with tf.name_scope("trainer"):
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

            # if the objective does not have its own gradients,
            # just use TF to do the derivative
            with tf.name_scope("gradient_collection"):
                differentiable_loss_sum = sum(
                    (o.weight if o.weight is not None else 1) * o.loss
                    for o in objectives
                    if o.gradients is None) + l1_cost + l2_cost
                implicit_gradients = self.optimizer.compute_gradients(
                    differentiable_loss_sum, self.var_list)

                # objectives that have their gradients explictly computed
                other_gradients = [
                    _scale_gradients(o.gradients, o.weight)
                    for o in objectives if o.gradients is not None]

                if other_gradients:
                    gradients = _sum_gradients(
                        [implicit_gradients] + other_gradients)
                else:
                    gradients = implicit_gradients


            # konstrukce stelarku
            existing_gradients, existing_vars = zip(*[
                (grad, var) for grad, var in gradients if grad is not None])

            with tf.variable_scope("gradient_buffer"):
                gradbufs = [tf.Variable(initial_value=tf.zeros_like(grad),
                                        trainable=False)
                            for grad in existing_gradients]

            with tf.variable_scope("loss_buffers"):
                objbufs = [tf.Variable(0.0, trainable=False)
                           for _ in objectives]
                diffbuf = tf.Variable(0.0, trainable=False)

            self.reset_ops = [tf.assign(gradbuf, tf.zeros_like(gradbuf))
                              for gradbuf in gradbufs]
            self.reset_ops.extend(tf.assign(objbuf, 0.0) for objbuf in objbufs)
            self.reset_ops.append(tf.assign(diffbuf, 0.0))

            self.accumulate_ops = [
                tf.assign_add(gradbuf, grad)
                for gradbuf, grad in zip(gradbufs, existing_gradients)]
            self.accumulate_ops.extend(
                tf.assign_add(objbuf, obj.loss)
                for objbuf, obj in zip(objbufs, objectives))
            self.accumulate_ops.append(
                tf.assign_add(diffbuf, differentiable_loss_sum))

            self.cumulator_counter = tf.Variable(
                0, trainable=False, name="self.cumulator_counter")
            self.reset_ops.append(tf.assign(self.cumulator_counter, 0))
            self.accumulate_ops.append(
                tf.assign_add(self.cumulator_counter, 1))

            averaged_grads = [grad / tf.to_float(self.cumulator_counter)
                              for grad in gradbufs]

            if clip_norm:
                assert clip_norm > 0.0
                averaged_grads = [tf.clip_by_norm(grad, clip_norm)
                                  for grad in averaged_grads]

            self.train_op = self.optimizer.apply_gradients(
                zip(averaged_grads, existing_vars), global_step=step)

            tf.summary.scalar("train_opt_cost",
                              diffbuf / tf.to_float(self.cumulator_counter),
                              collections=["summary_train"])

            # log all objectives
            for obj, objbuf in zip(objectives, objbufs):
                tf.summary.scalar(
                    obj.name, objbuf / tf.to_float(self.cumulator_counter),
                    collections=["summary_train"])

            for grad, var in zip(averaged_grads, existing_vars):
                if grad is not None:
                    tf.summary.histogram(
                        "gr_{}".format(var.name),
                        grad, collections=["summary_gradients"])

            ### TODO jeste je tu problem s tim, ze se skalary ani histogramy
            ### neukazujou v tensorboardu, když je batches_per_update > 1.

            self.histogram_summaries = tf.summary.merge(
                tf.get_collection("summary_gradients"))
            self.scalar_summaries = tf.summary.merge(
                tf.get_collection("summary_train"))

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
        self.result = None

        self.state = 0
        self.res_hist_sums = None
        self.res_scal_sums = None
        self.res_losses = None

    def next_to_execute(self) -> NextExecute:

        if self.state == 0:  # ACCUMULATING
            fetches = {"accumulators": self.trainer.accumulate_ops,
                       "counter": self.trainer.cumulator_counter,
                       "losses": self.trainer.losses}
            coders = self.trainer.all_coders

        elif self.state == 1:  # UPDATING
            fetches = {
                "train_op": self.trainer.train_op,
                "_update_ops": tf.get_collection(tf.GraphKeys.UPDATE_OPS)}

            if self.summaries:
                fetches["scalar_summaries"] = self.trainer.scalar_summaries
                fetches["histogram_summaries"] = self.trainer.histogram_summaries

            coders = []

        else:  # RESETTING
            fetches = {"resets": self.trainer.reset_ops}
            coders = []

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

        self.result = ExecutionResult(
            [], losses=self.res_losses,
            scalar_summaries=self.res_scal_sums,
            histogram_summaries=self.res_hist_sums,
            image_summaries=None)
