import tempfile
import subprocess
from typing import List

from neuralmonkey.logging import log


# pylint: disable=too-few-public-methods

class BeerWrapper(object):
    """Wrapper for BEER scorer.

    Paper: http://aclweb.org/anthology/D14-1025
    Code: https://github.com/stanojevic/beer
    """

    def __init__(self,
                 wrapper: str,
                 name: str="BEER",
                 encoding: str="utf-8") -> None:
        """Initialize the BEER wrapper.

        Args:
            name: Name of the evaluator.
            wrapper: Path to the BEER's executable.
            encoding: Data encoding.
        """
        self.wrapper = wrapper
        self.encoding = encoding
        self.name = name

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

            args = [self.wrapper, "-r", reffile.name, "-s", decfile.name]

            output_proc = subprocess.run(args,
                                         stderr=subprocess.PIPE,
                                         stdout=subprocess.PIPE)

            proc_stdout = output_proc.stdout.decode("utf-8")  # type: ignore
            lines = proc_stdout.splitlines()

            if len(lines) == 0:
                return 0.0

            try:
                beer_score = float(lines[0].split()[-1])
                return beer_score
            except IndexError:
                log("Error: Malformed output from BEER wrapper:", color="red")
                log(proc_stdout, color="red")
                log("=======", color="red")
                return 0.0
            except ValueError:
                log("Value error - beer '{}' is not a number.".format(
                    lines[0]), color="red")
                return 0.0
