# tests: lint, mypy

import numpy as np

# pylint: disable=too-few-public-methods


class Perplexity(object):

    def __init__(self, name="Perplexity"):
        self.name = name

    def __call__(self, perplexities, _):
        """
        Gets the average perplexity of a sentence. The list of perplexities is
        provided instead of the list of decoded sentences, the reference
        sentences are not needed at all.
        """

        return np.mean(perplexities)
