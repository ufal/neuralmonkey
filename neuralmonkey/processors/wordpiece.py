
from typing import List
import sys
import unicodedata
import six

from neuralmonkey.logging import log
from neuralmonkey.vocabulary import Vocabulary

# pylint: disable=too-few-public-methods

ALNUM_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


class WordpiecePreprocessor(object):
    """Loose implementation of the t2t SubwordTextTokenizer.

    Paper: TODO?
    Code: TODO
    """

    def __init__(self,
                 vocabulary: Vocabulary) -> None:
        log("Initializing wordpiece preprocessor")

        self.vocabulary = vocabulary

    def __call__(self, sentence: List[str]) -> List[str]:
        """See the code.

        TODO
        """

        sent = " ".join(sentence)
        tokens_str = sent[0]
        for i in range(1, len(sent)):
            if sent[i] == " ":
                continue

            if ((sent[i] in ALNUM_CHAR_SET) != (sent[i - 1] in ALNUM_CHAR_SET)
                    or sent[i - 1] == " "):
                tokens_str += " "
            tokens_str += sent[i]

        # Mark the end of each token
        # TODO: escape the characters properly
        tokens = [tok + "_" for tok in tokens_str.split(" ")]

        output = []  # type: List[str]
        for tok in tokens:
            tok_start = 0
            ret = []
            while tok_start < len(tok):
                for tok_end in range(len(tok), tok_start, -1):
                    subtoken = tok[tok_start:tok_end]
                    if subtoken in self.vocabulary:
                        ret.append(subtoken)
                        tok_start = tok_end
                        break
                else:
                    raise ValueError(
                        "Subword '{}' (from {}) is not in the vocabulary"
                        .format(tok[tok_start:len(tok)], tok))
            output += ret

        return output


class WordpiecePostprocessor(object):

    # pylint: disable=no-self-use
    def __init__(self) -> None:
        pass

    def __call__(self, decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [self.decode(s) for s in decoded_sentences]

    def decode(self, sentence: List[str]) -> List[str]:
        output = []
        for tok in sentence:
            if tok.endswith("_"):
                output.append(tok[:-1] + " ")
            else:
                output.append(tok)

        return "".join(output).rstrip(" ").split(" ")
