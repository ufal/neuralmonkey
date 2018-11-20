from typing import Any
from neuralmonkey.evaluators.evaluator import Evaluator, SequenceEvaluator


# pylint: disable=too-few-public-methods
# These classes are technically just a syntactic sugar.
class AccuracyEvaluator(SequenceEvaluator[Any]):
    """Accuracy Evaluator.

    This class uses the default `SequenceEvaluator` implementation, i.e. works
    on sequences of equal lengths (but can be used to others as well) and
    use `==` as the token scorer.
    """


class AccuracySeqLevelEvaluator(Evaluator[Any]):
    """Sequence-level accuracy evaluator.

    This class uses the default evaluator implementation. It gives 1.0 to equal
    sequences and 0.0 to others, averaging the scores over the batch.
    """


# pylint: disable=invalid-name
Accuracy = AccuracyEvaluator()
AccuracySeqLevel = AccuracySeqLevelEvaluator()
# pylint: enable=invalid-name
