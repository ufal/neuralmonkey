"""A set of helper functions for TensorFlow."""
from typing import Callable, Iterable, List, Optional, Tuple
# pylint: disable=unused-import
from typing import Dict, Set
# pylint: enable=unused-import
import numpy as np
import tensorflow as tf

from neuralmonkey.logging import debug, debug_enabled


def _get_current_experiment():
    # This is needed to avoid circular imports.
    from neuralmonkey.experiment import Experiment
    return Experiment.get_current()


def update_initializers(initializers: Iterable[Tuple[str, Callable]]) -> None:
    _get_current_experiment().update_initializers(initializers)


def get_initializer(var_name: str,
                    default: Callable = None) -> Optional[Callable]:
    """Return the initializer associated with the given variable name.

    The name of the current variable scope is prepended to the variable name.

    This should only be called during model building.
    """
    full_name = tf.get_variable_scope().name + "/" + var_name
    return _get_current_experiment().get_initializer(full_name, default)


def get_variable(name: str,
                 shape: List[Optional[int]] = None,
                 dtype: tf.DType = None,
                 initializer: Callable = None,
                 **kwargs) -> tf.Variable:
    """Get an existing variable with these parameters or create a new one.

    This is a wrapper around `tf.get_variable`. The `initializer` parameter is
    treated as a default which can be overriden by a call to
    `update_initializers`.

    This should only be called during model building.
    """
    return tf.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=get_initializer(name, initializer),
        **kwargs)


def tf_print(tensor: tf.Tensor,
             message: str = None,
             debug_label: str = None) -> tf.Tensor:
    """Print the value of a tensor to the debug log.

    Better than tf.Print, logs to console only when the "tensorval" debug
    subject is turned on.

    Idea found at: https://stackoverflow.com/a/39649614

    Args:
        tensor: The tensor whose value to print

    Returns:
        As tf.Print, this function returns a tensor identical to the input
        tensor, with the printing side-effect added.
    """
    def print_tensor(x: np.ndarray) -> tf.Tensor:
        if message is not None:
            debug(
                "{}, shape: {}:\n{}".format(message, x.shape, x), debug_label)
        else:
            debug("Shape: {}\n{}".format(x.shape, x), debug_label)
        return x

    # To save time, check if debug will print something
    if not debug_enabled(debug_label):
        return tensor

    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]

    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    return res
