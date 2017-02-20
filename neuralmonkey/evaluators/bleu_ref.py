import tempfile
from typing import List
import subprocess
from neuralmonkey.logging import log

# pylint: disable=too-few-public-methods
# to be further refactored


class BLEUReferenceImplWrapper(object):
    """Wrapper for TectoMT's wrapper for reference NIST and BLEU scorer"""

    def __init__(self, wrapper, name="BLEU", encoding="utf-8"):
        log("Reference BLEU wrapper is deprecated", color="red")
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

        reffile = tempfile.NamedTemporaryFile()
        reffile.write(ref_bytes)
        reffile.flush()

        output_proc = subprocess.run(["perl", self.wrapper, reffile.name],
                                     input=dec_bytes,
                                     stderr=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

        proc_stdout = output_proc.stdout.decode("utf-8")  # type: ignore
        lines = proc_stdout.splitlines()

        try:
            bleu_score = float(lines[0])
            return bleu_score
        except IndexError:
            log("Error: Malformed output from BLEU wrapper:", color="red")
            log(proc_stdout, color="red")
            log("=======", color="red")
            return 0.0
        except ValueError:
            log("Value error - bleu '{}' is not a number.".format(lines[0]),
                color="red")
            return 0.0
