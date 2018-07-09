from typing import NamedTuple, List
import tensorflow as tf


class AttentionLoopState(NamedTuple(
        "AttentionLoopState",
        [("contexts", tf.Tensor),
         ("weights", tf.Tensor)])):
    """Basic loop state of an attention mechanism.

    Attributes:
        contexts: A tensor of shape ``(query_time, batch, context_dim)`` which
            stores the context vectors for every decoder time step.
        weights: A tensor of shape ``(query_time, batch, keys_len)`` which
            stores the attention distribution over the keys given the query in
            each decoder time step.
    """


class HierarchicalLoopState(NamedTuple(
        "HierarchicalLoopState",
        [("child_loop_states", List),
         ("loop_state", AttentionLoopState)])):
    """Loop state of the hierarchical attention mechanism.

    The input to the hierarchical attetnion is the output of a set of
    underlying (child) attentions. To record the inner states of the underlying
    attentions, we use the ``HierarchicalLoopState``, which holds information
    about both the underlying attentions, and the top-level attention itself.

    Attributes:
        child_loop_states: A list of attention loop states of the underlying
            attention mechanisms.
        loop_state: The attention loop state of the top-level attention.
    """


class MultiHeadLoopState(NamedTuple(
        "MultiHeadLoopState",
        [("contexts", tf.Tensor),
         ("head_weights", List[tf.Tensor])])):
    """Loop state of a multi-head attention.

    Attributes:
        contexts: A tensor of shape ``(query_time, batch, context_dim)`` which
            stores the context vectors for every decoder time step.
        head_weights: A tensor of shape ``(query_time, n_heads, batch,
            keys_len)`` which stores the attention distribution over the keys
            given the query in each decoder time step **for each attention
            head**.
    """
