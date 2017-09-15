"""Module which implements decoding functions using multiple attentions
for RNN decoders.

See http://arxiv.org/abs/1606.07481

The attention mechanisms used in Neural Monkey are inherited from the
``BaseAttention`` class defined in this module.

Each attention object has the ``attention`` function which operates on the
``attention_states`` tensor.  The attention function receives the query tensor,
the decoder previous state and input, and its inner state, which could bear an
arbitrary structure of information. The default structure for this is the
``AttentionLoopState``, which contains a growing array of attention
distributions and context vectors in time. That's why there is the
``initial_loop_state`` function in the ``BaseAttention`` class.

Mainly for illustration purposes, the attention objects can keep their
*histories*, which is a dictionary populated with attention distributions in
time for every decoder, that used this attention object. This is because for
example the recurrent decoder is can be run twice for each sentence - once in
the *training* mode, in which the decoder gets the reference tokens on the
input, and once in the *running* mode, in which it gets its own outputs. The
histories object is constructed *after* the decoding and its construction
should be triggered manually from the decoder by calling the ``finalize_loop``
method.
"""
from typing import NamedTuple, Dict, Optional, Any, Tuple, Union

import tensorflow as tf

from neuralmonkey.model.stateful import TemporalStateful, SpatialStateful
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.decorators import tensor
from neuralmonkey.dataset import Dataset

# pylint: disable=invalid-name
AttentionLoopState = NamedTuple("AttentionLoopState",
                                [("contexts", tf.TensorArray),
                                 ("weights", tf.TensorArray)])
# pylint: enable=invalid-name


def empty_attention_loop_state() -> AttentionLoopState:
    """Create an empty attention loop state.

    The attention loop state is a technical object for storing the attention
    distributions and the context vectors in time. It is used with the
    ``tf.while_loop`` dynamic implementation of the decoder.

    This function returns an empty attention loop state which means there are
    two empty arrays, one for attention distributions in time, and one for
    the attention context vectors in time.
    """
    return AttentionLoopState(
        contexts=tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True,
            name="contexts"),
        weights=tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True,
            name="distributions"))


def get_attention_states(encoder: Union[TemporalStateful,
                                        SpatialStateful]) -> tf.Tensor:
    if isinstance(encoder, TemporalStateful):
        return encoder.temporal_states

    elif isinstance(encoder, SpatialStateful):
        # pylint: disable=no-member
        shape = [s.value for s in encoder.spatial_states.get_shape()[1:]]
        # pylint: enable=no-member
        return tf.reshape(encoder.spatial_states,
                          [-1, shape[0] * shape[1], shape[2]])
    else:
        raise AssertionError("Unknown encoder type")


def get_attention_mask(encoder: Union[TemporalStateful,
                                      SpatialStateful]) -> Optional[tf.Tensor]:
    if isinstance(encoder, TemporalStateful):
        if encoder.temporal_mask is None:
            raise ValueError("The encoder temporal mask should not be none")
        return encoder.temporal_mask

    elif isinstance(encoder, SpatialStateful):
        if encoder.spatial_mask is None:
            return None

        # pylint: disable=no-member
        shape = [s.value for s in encoder.spatial_mask.get_shape()[1:]]
        # pylint: enable=no-member
        return tf.reshape(encoder.spatial_mask, [-1, shape[0] * shape[1]])
    else:
        raise AssertionError("Unknown encoder type")


class BaseAttention(ModelPart):
    def __init__(self,
                 name: str,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)

        self.query_state_size = None  # type: tf.Tensor
        self._histories = {}  # type: Dict[str, tf.Tensor]

    @property
    def histories(self) -> Dict[str, tf.Tensor]:
        return self._histories

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: Any,
                  step: tf.Tensor) -> Tuple[tf.Tensor, Any]:
        """Get context vector for a given query."""
        raise NotImplementedError("Abstract method")

    def initial_loop_state(self) -> Any:
        """Get initial loop state for the attention object."""
        raise NotImplementedError("Abstract method")

    def finalize_loop(self, key: str, last_loop_state: Any) -> None:
        raise NotImplementedError("Abstract method")

    # pylint: disable=no-self-use
    @tensor
    def train_mode(self) -> tf.Tensor:
        return tf.placeholder(tf.bool, [], "train_mode")
    # pylint: enable=no-self-use

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        return {self.train_mode: train}

    @property
    def context_vector_size(self) -> int:
        raise NotImplementedError("Abstract property")
