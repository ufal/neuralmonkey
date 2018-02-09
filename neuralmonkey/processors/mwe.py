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
        pass

    def __call__(self, decoded_sentences: List[List[str]]) -> List[List[str]]:
        outputs = []
        for sentence in decoded_sentences:
            processed = []
            count = 0
            for tok in sentence:
                if tok == "CONT":
                    processed.append("{}".format(count))
                elif tok != "_":
                    count += 1
                    processed.append("{}:{}".format(count, tok))
                else:
                    processed.append(tok)
            outputs.append(processed)
        return outputs
