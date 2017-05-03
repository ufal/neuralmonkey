"""Training objective for self-critical learning.

Self-critic learning is a modifcation of the REINFORCE algorithm that uses the
reward of the train-time decoder output as a baselie in the update step.

For more details see: https://arxiv.org/pdf/1612.00563.pdf
"""

from typing import Callable, Iterable, Tuple
from itertools import takewhile
from collections import Counter

import numpy as np
import tensorflow as tf

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import END_TOKEN_INDEX


# pylint: disable=invalid-name
RewardFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
# pylint: enable=invalid-name


def reinforce_gradient(reward: tf.Tensor,
                       baseline: tf.Tensor,
                       decoded: tf.Tensor,
                       decoder: Decoder) -> tf.Tensor:
    """Gradients of loss w.r.t. decoder logits.

    This implements the central equation of the REINFORCE algorithm that
    estimates the gradients of the loss with respect to decoder logits. The
    ``stop_gradients`` function is applied on the gradients, such that it
    will not be further differentiated by TensorFlow.
    """

    reward_diff = tf.expand_dims(tf.expand_dims(reward - baseline, 0), 2)

    # runtime probabilities, shape (time, batch, vocab)
    runtime_probs = tf.nn.softmax(decoder.runtime_logits)
    runtime_decoded_onehot = tf.one_hot(decoded,
                                        len(decoder.vocabulary))

    # REINFORCE gradient, shape (time, batch, vocab)
    gradient = tf.stop_gradient(
        reward_diff * (runtime_probs - runtime_decoded_onehot))
    return tf.stop_gradient(gradient)


def self_critical_objective(decoder: Decoder,
                            reward_function: RewardFunction,
                            weight: float = None) -> Objective:

    # logits, shape (time, batch, vocab)
    train_logits = tf.stack(decoder.train_logits)
    runtime_logits = tf.stack(decoder.runtime_logits)
    runtime_mask = tf.stack(decoder.runtime_mask)

    # decoded, shape (time, batch)
    train_decoded = tf.argmax(train_logits, axis=2)
    runtime_decoded = tf.argmax(runtime_logits, axis=2)
    reference = decoder.train_inputs

    # rewards, shape (batch)
    train_reward = tf.py_func(
        reward_function, [reference, train_decoded], tf.float32)
    runtime_reward = tf.py_func(
        reward_function, [reference, runtime_decoded], tf.float32)

    tf.summary.scalar(
        "train_{}/{}".format(decoder.data_id, reward_function.__name__),
        tf.reduce_mean(runtime_reward),
        collections=["summary_train"])

    # REINFORCE gradient, shape (time, batch, vocab)
    reward_gradient = reinforce_gradient(
        runtime_reward, train_reward, runtime_decoded, decoder)

    # multiply the partial derivatives, shape (time, batch, vocab)
    # pylint: disable=invalid-unary-operand-type
    loss_matrix = (-reward_gradient * runtime_logits
                   * tf.expand_dims(tf.to_float(runtime_mask), 2))
    # pylint: enable=invalid-unary-operand-type

    # sum the matrix up (dot product of rows, sum over time, and over batch)
    cost = tf.reduce_sum(loss_matrix)

    return Objective(
        name="{}_self_critical".format(decoder.name),
        decoder=decoder,
        loss=cost,
        gradients=None,
        weight=weight)


def sentence_bleu(references: np.ndarray,
                  hypotheses: np.ndarray) -> np.ndarray:
    """Compute index-based sentence-level BLEU score.

    Computes sentence level BLEU on indices outputed by the decoder, i.e.
    whatever the decoder uses as a unit is used a token in the BLEU
    computation, ignoring the tokens may be sub-word units.
    """

    bleu_scores = []
    for ref, hyp in zip(np.transpose(references),
                        np.transpose(hypotheses)):
        matched_counts = []
        hyp_n_grams_counts = []

        for n in range(1, 5):
            matched, total, _ = _count_matching_n_grams(ref, hyp, n)

            if n > 1:
                matched += 1
                total += 1

            matched_counts.append(matched)
            hyp_n_grams_counts.append(total)

        precision = (
            np.prod(matched_counts) / np.prod(hyp_n_grams_counts)) ** .25
        ref_len = sum(1 for _ in
                      takewhile(lambda i: i != END_TOKEN_INDEX, ref))
        brevity_penalty = np.min([
            1., np.exp(1 - ref_len / hyp_n_grams_counts[0])])

        bleu_scores.append(brevity_penalty * precision)

    assert all(0 <= s <= 1 for s in bleu_scores)
    return np.ndarray(bleu_scores, dtype=np.float32)


def sentence_gleu(references: np.ndarray,
                  hypotheses: np.ndarray) -> np.ndarray:
    """Compute index-based GLEU score.

    GLEU score is a sentence-level metric used in Google's Neural MT as a
    reward in reinforcement learning (https://arxiv.org/abs/1609.08144).
    It is a minimum of precision and recall on 1- to 4-grams.

    It operates over the indices emitted by the decoder which are not
    necessarily tokens (could be characters or subword units).
    """
    gleu_scores = []

    for ref, hyp in zip(np.transpose(references),
                        np.transpose(hypotheses)):

        matched_counts = []
        hyp_n_grams_counts = []
        ref_n_grams_counts = []

        for n in range(1, 5):
            matched, total_hyp, total_ref = _count_matching_n_grams(
                ref, hyp, n)
            matched_counts.append(matched)
            hyp_n_grams_counts.append(total_hyp)
            ref_n_grams_counts.append(total_ref)

        precision = np.sum(matched_counts) / np.sum(hyp_n_grams_counts)
        recall = np.sum(matched_counts) / np.sum(ref_n_grams_counts)

        assert 0. <= precision <= 1.0
        assert 0. <= recall <= 1.0

        gleu_scores.append(min(precision, recall))

    return np.array(gleu_scores, dtype=np.float32)


def _count_matching_n_grams(ref: np.ndarray,
                            hyp: np.ndarray,
                            n: int) -> Tuple[int, int]:
    ref_counts = Counter()  # type: Counter[str]
    total_ref_n_grams = 0
    for n_gram in _get_n_grams(ref, n):
        ref_counts[str(n_gram)] += 1
        total_ref_n_grams += 1

    matched_n_grams = 0
    total_hyp_n_grams = 0
    hyp_n_grams = _get_n_grams(hyp, n)
    for n_gram in hyp_n_grams:
        n_gram_s = str(n_gram)
        if ref_counts[n_gram_s] > 0:
            matched_n_grams += 1
            ref_counts[n_gram_s] -= 1
        total_hyp_n_grams += 1

    assert matched_n_grams <= total_hyp_n_grams
    assert matched_n_grams <= total_ref_n_grams

    return matched_n_grams, total_hyp_n_grams, total_ref_n_grams


def _get_n_grams(indices: np.ndarray, order: int) -> Iterable[np.ndarray]:
    all_n_grams = [indices[i:i + order]
                   for i in range(len(indices) - order + 1)]
    return takewhile(lambda g: g[-1] != END_TOKEN_INDEX, all_n_grams)
