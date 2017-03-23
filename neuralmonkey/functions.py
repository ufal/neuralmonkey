from typing import Optional

import tensorflow as tf


def inverse_sigmoid_decay(param, rate, min_value: float=0.,
                          max_value: float=1.,
                          name: Optional[str]=None,
                          dtype=tf.float32) -> tf.Tensor:
    """Inverse sigmoid decay: k/(k+exp(x/k)).

    The result will be scaled to the range (min_value, max_value).

    Arguments:
        param: The parameter x from the formula.
        rate: Non-negative k from the formula.
    """

    with tf.name_scope(name, "InverseSigmoidDecay",
                       [rate, param, min_value, max_value]) as name:
        result = rate / (rate + tf.exp(param/rate))
        result = result * (max_value - min_value) + min_value
        result = tf.cast(result, dtype, name=name)

    return result


def piecewise_function(param, values, changepoints, name=None,
                       dtype=tf.float32):
    """A piecewise function.

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
                       [param, values, changepoints]) as name:
        values = [tf.convert_to_tensor(y, dtype=dtype) for y in values]
        # this is a trick to make each lambda return a different y:
        lambdas = [lambda y=y: y for y in values]
        predicates = [tf.less(param, x) for x in changepoints]
        return tf.case(list(zip(predicates, lambdas[:-1])), lambdas[-1],
                       name=name)
