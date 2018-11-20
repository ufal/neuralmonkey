# pylint: disable=too-few-public-methods, no-self-use, unused-argument
# This evaluator here is just an ugly hack to work with perplexity runner
from neuralmonkey.evaluators.evaluator import Evaluator


class AverageEvaluator(Evaluator[float]):
    """Just average the numeric output of a runner."""

    def score_instance(self, hypothesis: float, reference: float) -> float:
        return hypothesis
