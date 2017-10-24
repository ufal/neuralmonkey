"""Utilities.

This module contains helper functions that are suppoosed to be called from
the configuration file because calling the functions or the class constructors
directly would be inconvinent or impossible.
"""

from typing import Callable, TypeVar

import tensorflow as tf

from neuralmonkey.logging import warn

T = TypeVar("T")


def deprecated(func: Callable[..., T]) -> Callable[..., T]:
    def dep_func(*args, **kwargs) -> T:
        warn("Use of deprecated function from "
             + "'neuralmonkey.config.utils'. " +
             "Use '{}' instead.".format(func.__module__[13:]
                                        + "." + func.__name__))
        return func(*args, **kwargs)
    return dep_func


def adam_optimizer(learning_rate: float = 1e-4) -> tf.train.AdamOptimizer:
    return tf.train.AdamOptimizer(learning_rate)


def adadelta_optimizer(**kwargs) -> tf.train.AdadeltaOptimizer:
    return tf.train.AdadeltaOptimizer(**kwargs)


def variable(initial_value=0,
             trainable: bool = False, **kwargs) -> tf.Variable:
    return tf.Variable(initial_value, trainable, **kwargs)
