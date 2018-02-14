"""Collection of various functions and function wrappers."""

from typing import Optional

import math
import tensorflow as tf


def inverse_sigmoid_decay(param, rate, min_value: float = 0.,
                          max_value: float = 1.,
                          name: Optional[str] = None,
                          dtype=tf.float32) -> tf.Tensor:
    """Compute an inverse sigmoid decay: k/(k+exp(x/k)).

    The result will be scaled to the range (min_value, max_value).

    Arguments:
        param: The parameter x from the formula.
        rate: Non-negative k from the formula.
    """

    with tf.name_scope(name, "InverseSigmoidDecay",
                       [rate, param, min_value, max_value]) as s_name:
        result = rate / (rate + tf.exp(param / rate))
        result = result * (max_value - min_value) + min_value
        result = tf.cast(result, dtype, name=s_name)

    return result


def piecewise_function(param, values, changepoints, name=None,
                       dtype=tf.float32):
    """Compute a piecewise function.

    Arguments:
        param: The function parameter.
        values: List of function values (numbers or tensors).
        changepoints: Sorted list of points where the function changes from
            one value to the next. Must be one item shorter than `values`.
    """

    if len(changepoints) != len(values) - 1:
        raise ValueError("changepoints has length {}, expected {} (values "
                         "has length {})".format(len(changepoints),
                                                 len(values) - 1,
                                                 len(values)))

    with tf.name_scope(name, "PiecewiseFunction",
                       [param, values, changepoints]) as s_name:
        values = [tf.convert_to_tensor(y, dtype=dtype) for y in values]
        # this is a trick to make each lambda return a different y:
        lambdas = [lambda y=y: y for y in values]
        predicates = [tf.less(param, x) for x in changepoints]
        return tf.case(list(zip(predicates, lambdas[:-1])), lambdas[-1],
                       name=s_name)


def noam_decay(learning_rate: float,
               model_dimension: int,
               warmup_steps: int) -> tf.Tensor:
    """Return decay function as defined in Vaswani et al., 2017, Equation 3.

    https://arxiv.org/abs/1706.03762

    lrate = (d_model)^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)

    Arguments:
        model_dimension: Size of the hidden states of decoder and encoder
        warmup_steps: Number of warm-up steps
    """
    step = tf.to_float(tf.train.get_or_create_global_step())

    inv_sq_dim = 1 / math.sqrt(model_dimension)
    inv_sq3_warmup_steps = math.pow(warmup_steps, -1.5)

    inv_sq_step = 1 / tf.sqrt(step)
    warmup = step * inv_sq3_warmup_steps

    return learning_rate * inv_sq_dim * tf.minimum(inv_sq_step, warmup)
