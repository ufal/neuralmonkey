from typing import List
import rouge
from typeguard import check_argument_types
from neuralmonkey.evaluators.evaluator import Evaluator, check_lengths


# pylint: disable=too-few-public-methods
class RougeEvaluator(Evaluator[List[str]]):
    """Compute ROUGE score using third-party library."""

    def __init__(
            self, rouge_type: str,
            name: str = "ROUGE") -> None:
        check_argument_types()
        super().__init__(name)

        if rouge_type.lower() not in ["1", "2", "l"]:
            raise ValueError(
                ("Invalid type of rouge metric '{}', "
                 "must be '1', '2' or 'L'").format(rouge_type))

        self.rouge_type = rouge_type.lower()
        self.rouge = rouge.Rouge()

    @check_lengths
    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:
        hypotheses_str = [" ".join(l) for l in hypotheses]
        references_str = [" ".join(l) for l in references]

        rouge_res = self.rouge.get_scores(
            hypotheses_str, references_str, avg=True)

        rouge_value = rouge_res["rouge-{}".format(self.rouge_type)]["f"]

        return rouge_value


# pylint: disable=invalid-name
ROUGE_1 = RougeEvaluator("1", "ROUGE-1")
ROUGE_2 = RougeEvaluator("2", "ROUGE-2")
ROUGE_L = RougeEvaluator("l", "ROUGE-L")
# pylint: enable=invalid-name
