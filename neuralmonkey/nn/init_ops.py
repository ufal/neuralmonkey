import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import array_ops



def _assert_float_dtype(dtype):
  """Validate and return floating point type based on `dtype`.

  `dtype` must be a floating point type.

  Args:
    dtype: The data type to validate.

  Returns:
    Validated type.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype



def orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None):
  """Returns an initializer that generates an orthogonal matrix or a reshaped
  orthogonal matrix.

  If the shape of the tensor to initialize is two-dimensional, i is initialized
  with an orthogonal matrix obtained from the singular value decomposition of a
  matrix of uniform random numbers.

  If the shape of the tensor to initialize is more than two-dimensional, a matrix
  of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])` is initialized, where
  `n` is the length of the shape vector. The matrix is subsequently reshaped to
  give a tensor of the desired shape.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.

  Returns:
    An initializer that generates orthogonal tensors

  Raises:
    ValueError: if `dtype` is not a floating point type or if `shape` has fewer than two entries.
  """
  def _initializer(shape, dtype=_assert_float_dtype(dtype), partition_info=None):
    # Check the shape
    if len(shape) < 2:
      raise ValueError('the tensor to initialize must be at least two-dimensional')
    # Flatten the input shape with the last dimension remaining its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_rows, num_cols)

    # Generate a random matrix
    a = random_ops.random_uniform(flat_shape, dtype=dtype, seed=seed)
    # Compute the svd
    _, u, v = linalg_ops.svd(a, full_matrices=False)
    # Pick the appropriate singular value decomposition
    if num_rows > num_cols:
      q = u
    else:
      # Tensorflow departs from numpy conventions such that we need to transpose axes here
      q = array_ops.transpose(v)
    return gain * array_ops.reshape(q, shape)

  return _initializer
