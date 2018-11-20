from typing import Generic, TypeVar, List, Sequence
from functools import wraps
import numpy as np
from typeguard import check_argument_types

# pylint: disable=invalid-name
EvalType = TypeVar("EvalType")
SeqEvalType = TypeVar("SeqEvalType", bound=Sequence)
# pylint: enable=invalid-name


def check_lengths(scorer):
    @wraps(scorer)
    def decorate(self, hypotheses, references):
        if len(hypotheses) != len(references):
            raise ValueError("Hypothesis and reference lists do not have the "
                             "same length: {} vs {}.".format(len(hypotheses),
                                                             len(references)))
        if not hypotheses:
            raise ValueError("No hyp/ref pair to evaluate.")

        return scorer(self, hypotheses, references)
    return decorate


class Evaluator(Generic[EvalType]):
    """Base class for evaluators in Neural Monkey.

    Each evaluator has a `__call__` method which returns a score for a batch
    of model predictions given a the references. This class provides default
    implementations of `score_batch` and `score_instance` functions.
    """

    def __init__(self, name: str = None) -> None:
        check_argument_types()
        if name is None:
            name = type(self).__name__
            if name.endswith("Evaluator"):
                name = name[:-9]

        self._name = name

    @property
    def name(self) -> str:
        return self._name

    # pylint: disable=no-self-use
    # This function is meant to be overriden.
    def score_instance(self,
                       hypothesis: EvalType,
                       reference: EvalType) -> float:
        """Score a single hyp/ref pair.

        The default implementation of this method returns 1.0 when the
        hypothesis and the reference are equal and 0.0 otherwise.

        Arguments:
            hypothesis: The model prediction.
            reference: The golden output.

        Returns:
            A float.
        """
        if hypothesis == reference:
            return 1.0
        return 0.0
    # pylint: enable=no-self-use

    @check_lengths
    def score_batch(self,
                    hypotheses: List[EvalType],
                    references: List[EvalType]) -> float:
        """Score a batch of hyp/ref pairs.

        The default implementation of this method calls `score_instance` for
        each instance in the batch and returns the average score.

        Arguments:
            `hypotheses`: List of model predictions.
            `references`: List of golden outputs.

        Returns:
            A float.
        """
        return np.mean([self.score_instance(hyp, ref)
                        for hyp, ref in zip(hypotheses, references)])

    def __call__(self,
                 hypotheses: List[EvalType],
                 references: List[EvalType]) -> float:
        """Call the evaluator on a batch of data.

        By default, this function calls the `score_batch` method and returns
        the score.

        Arguments:
            `hypotheses`: List of model predictions.
            `references`: List of golden outputs.

        Returns:
            A float.
        """
        return self.score_batch(hypotheses, references)

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        """Compare scores using this evaluator.

        The default implementation regards the bigger score as better.

        Arguments:
            score1: The first score.
            score2: The second score.

        Returns
            An int. When `score1` is better, returns 1. When `score2` is
            better, returns -1. When the scores are equal, returns 0.
        """
        return (score1 > score2) - (score1 < score2)


class SequenceEvaluator(Evaluator[Sequence[EvalType]]):
    """Base class for token-level evaluators that work with sequences."""

    # pylint: disable=no-self-use
    # This method is supposed to be overriden.
    def score_token(self,
                    hyp_token: EvalType,
                    ref_token: EvalType) -> float:
        """Score a single hyp/ref pair of tokens.

        The default implementation returns 1.0 if the tokens are equal, 0.0
        otherwise.

        Arguments:
            hyp_token: A prediction token.
            ref_token: A golden token.

        Returns:
            A score for the token hyp/ref pair.
        """
        return float(hyp_token == ref_token)
    # pylint: enable=no-self-use

    @check_lengths
    def score_batch(self,
                    hypotheses: List[Sequence[EvalType]],
                    references: List[Sequence[EvalType]]) -> float:
        """Score batch of sequences.

        The default implementation assumes equal sequence lengths and operates
        on the token level (i.e. token-level scores from the whole batch are
        averaged (in contrast to averaging each sequence first)).

        Arguments:
            `hypotheses`: List of model predictions.
            `references`: List of golden outputs.

        Returns:
            A float.
        """
        token_scores = [self.score_token(h, r)
                        for hyp, ref in zip(hypotheses, references)
                        for h, r in zip(hyp, ref)]

        # All hypotheses empty - return zero score (needs to be here because of
        # the flattening)
        if not token_scores:
            return 0.0

        return np.mean(token_scores)
