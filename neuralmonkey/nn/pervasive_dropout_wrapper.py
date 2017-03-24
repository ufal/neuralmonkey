import tensorflow as tf

from neuralmonkey.checking import assert_shape


class PervasiveDropoutWrapper(tf.contrib.rnn.RNNCell):

    def __init__(self, cell, mask, scale) -> None:
        self._cell = cell
        self._mask = mask
        assert_shape(mask, [None, cell.sate_size])
        self._scale = scale

    @property
    def state_size(self) -> int:
        return self._cell.state_size

    @property
    def output_size(self) -> int:
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state)

        # self._mask is of shape [batch_size, state_size]
        # new_state is of shape [batch_size, state_size] (hopefully)
        new_state_dropped = new_state * self._scale * self._mask
        assert_shape(new_state_dropped, [None, self._cell.sate_size])
        return output, new_state_dropped
