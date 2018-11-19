import tempfile
import subprocess
from typing import List
from typeguard import check_argument_types

from neuralmonkey.logging import warn
from neuralmonkey.evaluators.evaluator import Evaluator


# pylint: disable=too-few-public-methods
class MultEvalWrapper(Evaluator[List[str]]):
    """Wrapper for mult-eval's reference BLEU and METEOR scorer."""

    def __init__(self,
                 wrapper: str,
                 name: str = "MultEval",
                 encoding: str = "utf-8",
                 metric: str = "bleu",
                 language: str = "en") -> None:
        """Initialize the wrapper.

        Arguments:
            wrapper: Path to multeval.sh script
            name: Name of the evaluator
            encoding: Encoding of input files
            language: Language of hypotheses and references
            metric: Evaluation metric "bleu", "ter", "meteor"
        """
        check_argument_types()
        super().__init__("{}_{}_{}".format(name, metric, language))

        self.wrapper = wrapper
        self.encoding = encoding
        self.language = language
        self.metric = metric

        if self.metric not in ["bleu", "ter", "meteor"]:
            warn("{} metric is not valid. Using bleu instead.".
                 format(self.metric))
            self.metric = "bleu"

    def score_batch(self,
                    hypotheses: List[List[str]],
                    references: List[List[str]]) -> float:

        ref_bytes = self.serialize_to_bytes(references)
        hyp_bytes = self.serialize_to_bytes(hypotheses)

        with tempfile.NamedTemporaryFile() as reffile, \
                tempfile.NamedTemporaryFile() as hypfile:

            reffile.write(ref_bytes)
            reffile.flush()

            hypfile.write(hyp_bytes)
            hypfile.flush()

            args = [self.wrapper, "eval", "--refs", reffile.name,
                    "--hyps-baseline", hypfile.name, "--metrics", self.metric]
            if self.metric == "meteor":
                args.extend(["--meteor.language", self.language])
                # problem: if meteor run for the first time,
                # paraphrase tables are downloaded

            output_proc = subprocess.run(
                args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            proc_stdout = output_proc.stdout.decode("utf-8")  # type: ignore
            lines = proc_stdout.splitlines()

            if not lines:
                return 0.0
            try:
                filtered = float(lines[1].split()[1])
                eval_score = filtered / 100.
                return eval_score
            except IndexError:
                warn("Error: Malformed output from MultEval wrapper:")
                warn(proc_stdout)
                warn("=======")
                return 0.0
            except ValueError:
                warn("Value error - '{}' is not a number.".format(lines[0]))
                return 0.0

    def serialize_to_bytes(self, sentences: List[List[str]]) -> bytes:
        joined = [" ".join(r) for r in sentences]
        string = "\n".join(joined) + "\n"
        return string.encode(self.encoding)
