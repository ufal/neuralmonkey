import re
from typing import List

from neuralmonkey.logging import log

# pylint: disable=too-few-public-methods


class MWELabelPreprocessor(object):
    """Wrapper class for TODO.

    Paper: TODO
    """

    def __init__(self) -> None:
        log("Initializing MWE label preprocessor")

    def __call__(self, sentence: List[str]) -> List[str]:
        output = []
        for word in sentence:
            word = re.sub(r"^[0-9]:", r"", word)
            word = re.sub(r";.*$", r"", word)
            word = re.sub(r"^[0-9]$", r"CONT", word)
            output.append(word)

        return output


class MWELabelPostprocessor(object):

    def __init__(self) -> None:
        self.count = 1

    def __call__(self, decoded_sentences: List[List[str]]) -> List[List[str]]:
        outputs = []
        for sentence in decoded_sentences:
            processed = []
            for tok in sentence:
                if tok == "CONT":
                    processed.append("{}".format(self.count))
                elif tok != "_":
                    processed.append("{}:{}".format(self.count, tok))
                    self.count += 1
                else:
                    processed.append(tok)
            outputs.append(processed)
            self.count = 1
        return outputs
