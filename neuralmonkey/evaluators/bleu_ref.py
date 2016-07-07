import tempfile
import subprocess

class BLEUReferenceImplWrapper(object):
    """Wrapper for TectoMT's wrapper for reference NIST and BLEU scorer"""

    def __init__(self, wrapper, name="BLEU", encoding="utf-8"):
        self.wrapper = wrapper
        self.encoding = encoding
        self.name = name


    def __call__(self, decoded, references):
        # type: (List[List[str]], List[List[str]]) -> float
        references_joined = [" ".join(r) for r in references]

        reffile = tempfile.NamedTemporaryFile()
        reffile.write("\n".join(references_joined).encode(self.encoding))
        reffile.flush()

        decoded_joined = [" ".join(d) for d in decoded]

        output_proc = subprocess.run(["perl", self.wrapper, reffile.name],
                                     input="\n".join(decoded_joined),
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)

        reffile.close()

        lines = output_proc.stdout.splitlines()
        return float(lines[1])
