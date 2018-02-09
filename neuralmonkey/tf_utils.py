"""A set of helper functions for TensorFlow."""
from typing import Callable, Iterable, List, Optional, Tuple
# pylint: disable=unused-import
from typing import Dict, Set
# pylint: enable=unused-import
import tensorflow as tf


# pylint: disable=invalid-name
current_experiment = None  # type: ignore
# pylint: enable=invalid-name


def update_initializers(initializers: Iterable[Tuple[str, Callable]]) -> None:
    if current_experiment:
        current_experiment.update_initializers(initializers)


def get_initializer(var_name: str,
                    default: Callable = None) -> Optional[Callable]:
    """Return the initializer associated with the given variable name.

    This should only be called during model building.
    """
    if current_experiment:
        return current_experiment.get_initializer(var_name, default)
    else:
        return default


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
