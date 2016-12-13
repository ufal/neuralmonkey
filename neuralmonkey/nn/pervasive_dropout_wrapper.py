import tensorflow as tf

# tests: lint, mypy


class PervasiveDropoutWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, mask, scale):
        self._cell = cell
        self._mask = mask
        self._scale = scale

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state)

        # self._mask is of shape [batch_size, state_size]
        # new_state is of shape [batch_size, state_size] (hopefully)
        new_state_dropped = new_state * self._scale * self._mask
        return output, new_state_dropped
