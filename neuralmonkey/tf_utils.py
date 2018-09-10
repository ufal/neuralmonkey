"""A set of helper functions for TensorFlow."""
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from neuralmonkey.logging import debug, debug_enabled

# pylint: disable=invalid-name
ShapeSpec = List[int]
# pylint: enable=invalid-name


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
                 shape: ShapeSpec = None,
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


def get_shape_list(x: tf.Tensor) -> List[Union[int, tf.Tensor]]:
    """Return list of dims, statically where possible.

    Compute the static shape of a tensor. Where the dimension is not static
    (e.g. batch or time dimension), symbolic Tensor is returned.

    Based on tensor2tensor.

    Arguments:
        x: The ``Tensor`` to process.

    Returns:
        A list of integers and Tensors.
    """
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def get_state_shape_invariants(state: tf.Tensor) -> tf.TensorShape:
    """Return the shape invariant of a tensor.

    This function computes the loosened shape invariant of a state tensor.
    Only invariant dimension is the state size dimension, which is the last.

    Based on tensor2tensor.

    Arguments:
        state: The state tensor.

    Returns:
        A ``TensorShape`` object with all but the last dimensions set to
        ``None``.
    """
    shape = state.shape.as_list()
    for i in range(0, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def gather_flat(x: tf.Tensor,
                indices: tf.Tensor,
                batch_size: Union[int, tf.Tensor] = 1,
                beam_size: Union[int, tf.Tensor] = 1) -> tf.Tensor:
    """Gather values from the flattened (shape=[batch * beam, ...]) input.

    This function expects a flattened tensor with first dimension of size
    *batch x beam* elements. Using the given batch and beam size, it reshapes
    the input tensor to a tensor of shape ``(batch, beam, ...)`` and gather
    the values from it using the index tensor.

    Arguments:
        x: A flattened ``Tensor`` from which to gather values.
        indices: Index tensor.
        batch_size: The size of the batch.
        beam_size: The size of the beam.

    Returns:
        The ``Tensor`` of gathered values.
    """
    if x.shape.ndims == 0:
        return x

    shape = [batch_size, beam_size] + get_shape_list(x)[1:]
    gathered = tf.gather_nd(tf.reshape(x, shape), indices)
    return tf.reshape(gathered, [-1] + shape[2:])


def partial_transpose(x: tf.Tensor, indices: List[int]) -> tf.Tensor:
    """Do a transpose on a subset of tensor dimensions.

    Compute a permutation of first k dimensions of a tensor.

    Arguments:
        x: The ``Tensor`` to transpose.
        indices: The permutation of the first k dimensions of ``x``.

    Returns:
        The transposed tensor.
    """
    dims = x.shape.ndims
    orig_indices = list(range(dims))

    return tf.transpose(x, indices + orig_indices[len(indices):])


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


def layer_norm(x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
    """Layer normalize the tensor x, averaging over the last dimension.

    Implementation based on tensor2tensor.

    Arguments:
        x: The ``Tensor`` to normalize.
        epsilon: The smoothing parameter of the normalization.

    Returns:
        The normalized tensor.
    """
    with tf.variable_scope("LayerNorm"):
        gamma = get_variable(
            name="gamma",
            shape=[x.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        beta = get_variable(
            name="beta",
            shape=[x.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean),
            axis=[-1],
            keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * gamma + beta


def append_tensor(tensor: tf.Tensor, appendval: tf.Tensor) -> tf.Tensor:
    """Append an ``N``-D Tensor to an ``(N+1)``-D Tensor.

    Arguments:
        tensor: The original Tensor
        appendval: The Tensor to add

    Returns:
        An ``(N+1)``-D Tensor with ``appendval`` on the last position.
    """
    return tf.concat([tensor, tf.expand_dims(appendval, 0)], 0)
