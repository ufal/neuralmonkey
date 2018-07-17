from typing import List
import rouge


class RougeEvaluator:
    """Compute ROUGE score using third-party library."""

    # pylint: disable=too-few-public-methods

    def __init__(
            self, rouge_type: str,
            name: str = "ROUGE") -> None:

        if rouge_type.lower() not in ["1", "2", "l"]:
            raise ValueError(
                ("Invalid type of rouge metric '{}', "
                 "must be '1', '2' or 'L'").format(rouge_type))

        self.name = name
        self.rouge_type = rouge_type.lower()
        self.rouge = rouge.Rouge()

    def __call__(self,
                 decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        decoded_str = [" ".join(l) for l in decoded]
        references_str = [" ".join(l) for l in references]

        rouge_res = self.rouge.get_scores(
            decoded_str, references_str, avg=True)

        rouge_value = rouge_res["rouge-{}".format(self.rouge_type)]["f"]

        return rouge_value

    @staticmethod
    def compare_scores(score1: float, score2: float) -> int:
        # the bigger the better
        return (score1 > score2) - (score1 < score2)


# pylint: disable=invalid-name
ROUGE_1 = RougeEvaluator("1", "ROUGE-1")
ROUGE_2 = RougeEvaluator("2", "ROUGE-2")
ROUGE_L = RougeEvaluator("l", "ROUGE-L")
# pylint: enable=invalid-name
