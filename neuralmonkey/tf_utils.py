"""A set of helper functions for TensorFlow."""
from typing import Callable, Iterable, List, Optional, Tuple
# pylint: disable=unused-import
from typing import Dict, Set
# pylint: enable=unused-import
import tensorflow as tf

from neuralmonkey.logging import debug


# pylint: disable=invalid-name
_initializers = {}  # type: Dict[str, Callable]
_initialized_variables = set()  # type: Set[str]
# pylint: enable=invalid-name


def update_initializers(initializers: Iterable[Tuple[str, Callable]]) -> None:
    _initializers.update(initializers)


def get_initializer(var_name: str,
                    default: Callable = None) -> Optional[Callable]:
    """Return the initializer associated with the given variable name."""
    full_name = tf.get_variable_scope().name + "/" + var_name
    initializer = _initializers.get(full_name, default)
    if initializer is not default:
        debug("Using {} for variable {}".format(initializer, full_name))
    _initialized_variables.add(full_name)
    return initializer


def get_unused_initializers() -> List[str]:
    """Return the names of unused initializers."""
    return [name for name in _initializers
            if name not in _initialized_variables]


def get_variable(name: str,
                 shape: List[Optional[int]] = None,
                 dtype: tf.DType = None,
                 initializer: Callable = None,
                 **kwargs) -> tf.Variable:
    """Get an existing variable with these parameters or create a new one.

    This is a wrapper around `tf.get_variable`. The `initializer` parameter is
    treated as a default which can be overriden by a call to
    `update_initializers`.
    """
    return tf.get_variable(
        name=name, shape=shape, dtype=dtype,
        initializer=get_initializer(name, initializer),
        **kwargs)
