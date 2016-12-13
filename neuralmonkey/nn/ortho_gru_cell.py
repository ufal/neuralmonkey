import tensorflow as tf

from neuralmonkey.nn.init_ops import orthogonal_initializer

# tests: lint, mypy


class OrthoGRUCell(tf.nn.rnn_cell.GRUCell):
    """Classic GRU cell but initialized using random orthogonal matrices"""

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "OrthoGRUCell") as vscope:
            vscope.set_initializer(orthogonal_initializer())
            return super().__call__(inputs, state, vscope)
