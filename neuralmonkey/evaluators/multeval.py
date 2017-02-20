import tempfile
import subprocess
from typing import List

from neuralmonkey.logging import log


# pylint: disable=too-few-public-methods


class MultEvalWrapper(object):
    """Wrapper for mult-eval's reference BLEU and METEOR scorer."""

    def __init__(self, wrapper: str, name: str="MultEval",
                 encoding: str="utf-8",
                 metric: str="bleu", language: str="en") -> None:
        """
        :param wrapper: path to multeval.sh script
        :param name: name of the evaluator
        :param encoding: encoding of input files
        :param language: language of hypotheses and references
        :param metric: evaluation metric "bleu", "ter", "meteor"
        """
        self.wrapper = wrapper
        self.encoding = encoding
        self.name = "{}_{}_{}".format(name, metric, language)
        self.language = language
        self.metric = metric

        if self.metric not in ["bleu", "ter", "meteor"]:
            log("{} metric is not valid. Using bleu instead.".
                format(self.metric), color="red")
            self.metric = "bleu"

    def serialize_to_bytes(self, sentences: List[List[str]]) -> bytes:
        joined = [" ".join(r) for r in sentences]
        string = "\n".join(joined) + "\n"
        return string.encode(self.encoding)

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:

        ref_bytes = self.serialize_to_bytes(references)
        dec_bytes = self.serialize_to_bytes(decoded)

        with tempfile.NamedTemporaryFile() as reffile, \
                tempfile.NamedTemporaryFile() as decfile:

            reffile.write(ref_bytes)
            reffile.flush()

            decfile.write(dec_bytes)
            decfile.flush()

            args = [self.wrapper, "eval", "--refs", reffile.name,
                    "--hyps-baseline", decfile.name, "--metrics", self.metric]
            if self.metric == "meteor":
                args.extend(["--meteor.language", self.language])
                # problem: if meteor run for the first time,
                # paraphrase tables are downloaded

            output_proc = subprocess.run(args,
                                         stderr=subprocess.PIPE,
                                         stdout=subprocess.PIPE)

            proc_stdout = output_proc.stdout.decode("utf-8")  # type: ignore
            lines = proc_stdout.splitlines()

            if len(lines) == 0:
                return 0.0
            try:
                filtered = float(lines[1].split()[1])
                eval_score = filtered/100.
                return eval_score
            except IndexError:
                log("Error: Malformed output from MultEval wrapper:",
                    color="red")
                log(proc_stdout, color="red")
                log("=======", color="red")
                return 0.0
            except ValueError:
                log("Value error - '{}' is not a number.".format(lines[0]),
                    color="red")
                return 0.0
