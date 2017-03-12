import tensorflow as tf


# pylint: disable=too-few-public-methods
class OrthoGRUCell(tf.contrib.rnn.GRUCell):
    """Classic GRU cell but initialized using random orthogonal matrices"""

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "OrthoGRUCell") as vscope:
            vscope.set_initializer(tf.orthogonal_initializer())
            return super().__call__(inputs, state, vscope)
