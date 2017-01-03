from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import re

import tensorflow as tf

from neuralmonkey.runners.base_runner import (collect_encoders, Executable,
                                              ExecutionResult, NextExecute)

# tests: lint, mypy

# pylint: disable=invalid-name
Gradients = List[Tuple[tf.Tensor, tf.Variable]]
ObjectiveWeight = Union[tf.Tensor, float, None]
Objective = NamedTuple('Objective',
                       [('name', str),
                        ('decoder', Any),
                        ('loss', tf.Tensor),
                        ('gradients', Optional[Gradients]),
                        ('weight', ObjectiveWeight)])

BIAS_REGEX = re.compile(r'[Bb]ias')


# pylint: disable=too-few-public-methods,too-many-locals
class GenericTrainer(object):

    def __init__(self, objectives: List[Objective],
                 l1_weight=0.0, l2_weight=0.0,
                 clip_norm=False, optimizer=None, global_step=None) -> None:

        self.optimizer = optimizer or tf.train.AdamOptimizer(1e-4)

        with tf.variable_scope('regularization'):
            regularizable = [v for v in tf.trainable_variables()
                             if BIAS_REGEX.findall(v.name)]
            l1_value = sum(tf.reduce_sum(abs(v)) for v in regularizable)
            l1_cost = l1_weight * l1_value if l1_weight > 0 else 0.0

            l2_value = sum(tf.reduce_sum(v ** 2) for v in regularizable)
            l2_cost = l2_weight * l2_value if l2_weight > 0 else 0.0

        # unweighted losses for fetching
        self.losses = [o.loss for o in objectives] + [l1_value, l2_value]
        tf.scalar_summary('train_l1', l1_value, collections=["summary_train"])
        tf.scalar_summary('train_l2', l2_value, collections=["summary_train"])

        # if the objective does not have its own gradients,
        # just use TF to do the derivative
        differentiable_loss_sum = sum(
            (o.weight if o.weight is not None else 1) * o.loss
            for o in objectives if o.gradients is None) + l1_cost + l2_cost
        implicit_gradients = self._get_gradients(differentiable_loss_sum)

        # objectives that have their gradients explictly computed
        other_gradients = [
            _scale_gradients(o.gradients, o.weight)
            for o in objectives if o.gradients is not None]

        if other_gradients:
            gradients = _sum_gradients([implicit_gradients] + other_gradients)
        else:
            gradients = implicit_gradients

        if clip_norm:
            assert clip_norm > 0.0
            gradients = [(tf.clip_by_norm(grad, clip_norm), var)
                         for grad, var in gradients]

        self.all_coders = set.union(*(collect_encoders(obj.decoder)
                                      for obj in objectives))

        if global_step is None:
            global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_step = global_step

        self.train_op = self.optimizer.apply_gradients(
            gradients, global_step=self.global_step)

        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary('gr_' + var.name,
                                     grad, collections=["summary_gradients"])

        self.histogram_summaries = tf.merge_summary(
            tf.get_collection("summary_gradients"))
        self.scalar_summaries = tf.merge_summary(
            tf.get_collection("summary_train"))

    def _get_gradients(self, tensor: tf.Tensor) -> Gradients:
        gradient_list = self.optimizer.compute_gradients(tensor)
        return gradient_list

    # pylint: disable=unused-argument
    def get_executable(self, train=False, summaries=True) -> Executable:
        return TrainExecutable(self.all_coders,
                               self.train_op,
                               self.losses,
                               self.scalar_summaries if summaries else None,
                               self.histogram_summaries if summaries else None)


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

    def __init__(self, all_coders, train_op, losses, scalar_summaries,
                 histogram_summaries):
        self.all_coders = all_coders
        self.train_op = train_op
        self.losses = losses
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries

        self.result = None

    def next_to_execute(self) -> NextExecute:
        fetches = {'train_op': self.train_op}
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries
        fetches['losses'] = self.losses

        return self.all_coders, fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        if self.scalar_summaries is None:
            scalar_summaries = None
            histogram_summaries = None
        else:
            # TODO collect summaries from different sessions
            scalar_summaries = results[0]['scalar_summaries']
            histogram_summaries = results[0]['histogram_summaries']

        losses_sum = [0. for _ in self.losses]
        for session_result in results:
            for i in range(len(self.losses)):
                # from the end, losses are last ones
                losses_sum[i] += session_result['losses'][i]
        avg_losses = [s / len(results) for s in losses_sum]

        self.result = ExecutionResult(
            [], losses=avg_losses,
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)
