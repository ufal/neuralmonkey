"""Decoding functions using multiple attentions for RNN decoders.

See http://arxiv.org/abs/1606.07481

The attention mechanisms used in Neural Monkey are inherited from the
``BaseAttention`` class defined in this module.

The attention function can be viewed as a soft lookup over an associative
memory. The *query* vector is used to compute a similarity score of the *keys*
of the associative memory and the resulting scores are used as weights in a
weighted sum of the *values* associated with the keys. We call the
(unnormalized) similarity scores *energies*, we call *attention distribution*
the energies after (softmax) normalization, and we call the resulting
weighted sum of states a *context vector*.

Note that it is possible (and true in most cases) that the attention keys
are equal to the values. In case of self-attention, even queries are from the
same set of vectors.

To abstract over different flavors of attention mechanism, we conceptualize the
procedure as follows: Each attention object has the ``attention`` function
which operates on the query tensor. The attention function receives the query
tensor (the decoder state) and optionally the previous state of the decoder,
and computes the context vector. The function also receives a *loop state*,
which is used to store data in an autoregressive loop that generates a
sequence.

The attention uses the loop state to store to store attention distributions
and context vectors in time. This structure is called ``AttentionLoopState``.
To be able to initialize the loop state, each attention object that uses this
feature defines the ``initial_loop_state`` function with empty tensors.

Since there can be many *modes* in which the decoder that uses the attention
operates, the attention objects have the ``finalize_loop`` method, which takes
the last attention loop state and the name of the mode (a string) and processes
this data to be available in the ``histories`` dictionary. The single and most
used example of two *modes* are the *train* and *runtime* modes of the
autoregressive decoder.
"""
from typing import Dict, Optional, Any, Tuple, Union

import tensorflow as tf

from neuralmonkey.attention.namedtuples import AttentionLoopState
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.stateful import TemporalStateful, SpatialStateful

# pylint: disable=invalid-name
Attendable = Union[TemporalStateful, SpatialStateful]
# pylint: enable=invalid-name


def empty_attention_loop_state(
        batch_size: Union[int, tf.Tensor],
        length: Union[int, tf.Tensor],
        dimension: Union[int, tf.Tensor]) -> AttentionLoopState:
    """Create an empty attention loop state.

    The attention loop state is a technical object for storing the attention
    distributions and the context vectors in time. It is used with the
    ``tf.while_loop`` dynamic implementation of decoders.

    Arguments:
        batch_size: The size of the batch.
        length: The number of encoder states (keys).
        dimension: The dimension of the context vector

    Returns:
        This function returns an empty attention loop state which means
        there are two empty Tensors one for attention distributions in time,
        and one for the attention context vectors in time.
    """
    return AttentionLoopState(
        contexts=tf.zeros(shape=[0, batch_size, dimension], name="contexts"),
        weights=tf.zeros(shape=[0, batch_size, length], name="distributions"))


def get_attention_states(encoder: Attendable) -> tf.Tensor:
    """Return the temporal or spatial states of an encoder.

    Arguments:
        encoder: The encoder with the states to attend.

    Returns:
        Either a 3D or a 4D tensor, depending on whether the encoder is
        temporal (e.g. recurrent encoder) or spatial (e.g. a CNN encoder).
        The first two dimensions are (batch, time).
    """
    if isinstance(encoder, TemporalStateful):
        return encoder.temporal_states

    if isinstance(encoder, SpatialStateful):
        shape = encoder.spatial_states.get_shape().as_list()
        return tf.reshape(encoder.spatial_states,
                          [-1, shape[1] * shape[2], shape[3]])

    raise TypeError("Unknown encoder type")


def get_attention_mask(encoder: Attendable) -> Optional[tf.Tensor]:
    """Return the temporal or spatial mask of an encoder.

    Arguments:
        encoder: The encoder to get the mask from.

    Returns:
        Either a 2D or a 3D tensor, depending on whether the encoder is
        temporal (e.g. recurrent encoder) or spatial (e.g. a CNN encoder).
    """
    if isinstance(encoder, TemporalStateful):
        if encoder.temporal_mask is None:
            raise ValueError("The encoder temporal mask should not be none")
        return encoder.temporal_mask

    if isinstance(encoder, SpatialStateful):
        if encoder.spatial_mask is None:
            return None
        shape = encoder.spatial_states.get_shape().as_list()
        return tf.reshape(encoder.spatial_mask, [-1, shape[1] * shape[2]])

    raise TypeError("Unknown encoder type")


class BaseAttention(ModelPart):
    """The abstract class for the attenion mechanism flavors."""

    def __init__(self,
                 name: str,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        """Create a new ``BaseAttention`` object."""
        ModelPart.__init__(
            self, name, reuse, save_checkpoint, load_checkpoint, initializers)

        self.query_state_size = None  # type: tf.Tensor
        self._histories = {}  # type: Dict[str, tf.Tensor]

    @property
    def histories(self) -> Dict[str, tf.Tensor]:
        """Return the attention histories dictionary.

        Use this property after it has been populated.

        Returns:
            The attention histories dictionary.
        """
        return self._histories

    def attention(self,
                  query: tf.Tensor,
                  decoder_prev_state: tf.Tensor,
                  decoder_input: tf.Tensor,
                  loop_state: Any) -> Tuple[tf.Tensor, Any]:
        """Get context vector for a given query."""
        raise NotImplementedError("Abstract method")

    def initial_loop_state(self) -> Any:
        """Get initial loop state for the attention object.

        Returns:
            The newly created initial loop state object.
        """
        raise NotImplementedError("Abstract method")

    def finalize_loop(self, key: str, last_loop_state: Any) -> None:
        """Store the attention histories from loop state under a given key.

        Arguments:
            key: The key to the histories dictionary to store the data in.
            last_loop_state: The loop state object from the last state of
                the decoding loop.
        """
        raise NotImplementedError("Abstract method")

    @property
    def context_vector_size(self) -> int:
        """Return the static size of the context vector.

        Returns:
            An integer specifying the context vector dimension.
        """
        raise NotImplementedError("Abstract property")

    def visualize_attention(self, key: str, max_outputs: int = 16) -> None:
        """Include the attention histories under a given key into a summary.

        Arguments:
            key: The key to the attention histories dictionary.
            max_outputs: Maximum number of images to save.
        """
        if key not in self.histories:
            raise KeyError(
                "Key {} not among attention histories".format(key))

        alignments = tf.expand_dims(
            tf.transpose(self.histories[key], perm=[1, 2, 0]), -1)

        summary_name = "{}.{}".format(self.name, key)

        tf.summary.image(
            summary_name, alignments, collections=["summary_att_plots"],
            max_outputs=max_outputs)
