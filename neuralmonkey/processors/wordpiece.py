"""Loose reimplementation of the t2t tokenizer.

Original code:
https://github.com/tensorflow/tensor2tensor/blob/v1.5.5/tensor2tensor/data_generators/tokenizer.py

Provides a WordpiecePreprocessor, a higher order function which takes a
vocabulary object and returns a preprocessor, and a WordpiecePostprocessor.

Note that the latter is not a higher order function and can be used directly
without making a new section in the configuration.
"""
from typing import List, Callable, Set
import re

from typeguard import check_argument_types
from neuralmonkey.processors.helpers import pyfunc_wrapper
from neuralmonkey.vocabulary import Vocabulary


UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")


def escape_token(token: str, alphabet: Set[str]) -> str:
    """Escapes the token in the t2t fashion.

    Underscores are regarded as an end of a token, so they must be escaped.
    Additionally, they/we escape also the OOA (out-of-alphabet) characters
    using their unicode code.
    """

    esc_token = token.replace("\\", "\\\\")  # replace 1 backslash with 2
    esc_token = esc_token.replace("_", "\\u")  # replace underscore with "\u"

    # replace OOA symbol `s` with \1234; where 1234 is `ord(s)`
    characters = [c if c in alphabet and c != "\n" else "\\{};".format(ord(c))
                  for c in token]  # not sure about the "\n"-part

    return "".join(characters) + "_"


def unescape_token(escaped_token: str) -> str:
    """Inverse function for escape_token."""

    # Ends with underscore -> remove it
    token = escaped_token
    token = token[:-1] if token.endswith("_") else token

    def match(m):
        if m.group(1) is None:
            return "_" if m.group(0) == "\\u" else "\\"

        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return u"\u3013"  # Unicode for undefined character.

    # The substitution works because of the left-to-right nature of matching
    return UNESCAPE_REGEX.sub(match, token)


def wordpiece_encode(sentence: List[str], vocabulary: Vocabulary) -> List[str]:
    """Convert tokens to subtokens using a vocabulary of subtokens.

    A greedy implementation, as in t2t referenced above.

    We search for the longest subtoken available in the vocabulary from left to
    right.
    """
    tokens = []
    for token in sentence:
        esc_token = escape_token(token, vocabulary.alphabet)

        subtokens = []
        current_subtoken_start = 0
        token_len = len(esc_token)

        while current_subtoken_start < len(esc_token):

            # TODO: they optimize this by ranging from
            # min(token_len, max_subtoken_len + start)
            # this can be achieved by saving the len of longest word in vocab
            for end in range(token_len, current_subtoken_start, -1):
                subtoken = esc_token[current_subtoken_start:end]

                if subtoken in vocabulary:
                    subtokens.append(subtoken)
                    current_subtoken_start = end
                    break
            else:  # executed if the loop is not exited by the break statement
                raise AssertionError(
                    "No token substring found in the vocab ({})."
                    .format(esc_token[current_subtoken_start:]))

        # TODO: they also optimize this by caching the segmentation of the
        # escaped tokens.
        tokens.extend(subtokens)
    return tokens


def wordpiece_decode(sentence: List[str]) -> List[str]:
    """Postprocess the wordpieces into a sentence.

    First, retokenize the sentence - join and split around underscores.
    Second, unescape tokens throwing away any empty tokens encountered.
    """
    retokenized = "".join(sentence).split("_")
    unescaped = [unescape_token(tok) for tok in retokenized if tok]
    return [tok for tok in unescaped if tok]


def wordpiece_decode_batch(sentences: List[List[str]]) -> List[List[str]]:
    return [wordpiece_decode(s) for s in sentences]


def get_wordpiece_preprocessor(
        vocabulary: Vocabulary) -> Callable[[List[str]], List[str]]:
    check_argument_types()

    @pyfunc_wrapper
    def preprocessor(sentence: List[str]) -> List[str]:
        return wordpiece_encode(sentence, vocabulary)

    return preprocessor


# pylint: disable=invalid-name
# Syntactic sugar for configuration
WordpiecePreprocessor = get_wordpiece_preprocessor
WordpiecePostprocessor = wordpiece_decode_batch
