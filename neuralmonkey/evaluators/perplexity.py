import numpy as np

class Perplexity(object):
    #pylint disable=(too-few-public-methods)

    def __init__(name="Perplexity"):
        self.name = name


    def __call__(perplexities, _):
        """
        Gets the average perplexity of a sentence. The list of perplexities is
        provided instead of the list of decoded sentences, the reference sentences
        are not needed at all.
        """
        return np.mean(perplexities)
