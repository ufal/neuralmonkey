"""Collection of helper functions providing optimizers for trainers"""
from typing import Union, Callable, Optional, Any

import tensorflow as tf
from typeguard import check_argument_types

# pylint: disable=invalid-name
LearningRate = Union[float, tf.Tensor]
OptimizerGetter = Union[
    Callable[[LearningRate], tf.train.Optimizer],
    Callable[[], tf.train.Optimizer]]
# pylint: enable=invalid-name


def adam_optimizer(
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        use_locking: bool = False) -> OptimizerGetter:
    check_argument_types()
    def get_optimizer(
            learning_rate: LearningRate = learning_rate) -> tf.train.Optimizer:
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2,
                                      epsilon=epsilon, use_locking=use_locking)
    return get_optimizer

def lazy_adam_optimizer(
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        use_locking: bool = False) -> OptimizerGetter:
    check_argument_types()
    def get_optimizer(
            learning_rate: LearningRate = learning_rate) -> tf.train.Optimizer:
        return tf.contrib.opt.LazyAdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
            use_locking=use_locking)
    return get_optimizer

def adadelta_optimizer(
        learning_rate: float = 0.001,
        rho: float = 0.95,
        epsilon: float = 1e-08,
        use_locking: bool = False) -> OptimizerGetter:
    check_argument_types()
    def get_optimizer(
            learning_rate: LearningRate = learning_rate) -> tf.train.Optimizer:
        return tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate, rho=rho, epsilon=epsilon,
            use_locking=use_locking)
    return get_optimizer


def variable(initial_value: Any = 0,
             trainable: bool = False, **kwargs) -> tf.Variable:
    check_argument_types()
    return tf.Variable(initial_value, trainable, **kwargs)
