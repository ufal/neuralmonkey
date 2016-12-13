"""
This module gathers some tools for image preprocessing.
"""

# tests: lint, mypy
# pylint: skip-file

import numpy as np
from scipy.misc import imread, imresize

# pylint: disable=too-few-public-methods


class STRPreprocessor(object):
    """
    This class implements a function that preprocesses an image for scene text
    recognition. It normalizes the height and pads/shortens it to the
    reqreusted width.
    """

    def __init__(self, height, max_width):
        self.height = height
        self.max_width = max_width

        self.shrinkages = []
        self.paddings = []

    def __call__(self, path):
        """
        Path is either path to the image or a list list conating the path (in
        case the path has been loaded as a tokenized text).
        """

        if isinstance(path, list):
            path = " ".join(path)

        img = imread(path) / 255.0

        if img.shape[0] != self.height:
            ratio = float(self.height) / img.shape[0]
            width = int(ratio * img.shape[1])
            img = imresize(img, (self.height, width))

        if img.shape[1] >= self.max_width:
            self.shrinkages.append(float(img.shape[1] - self.max_width))
            return img[:, :self.max_width]
        else:
            rest = self.max_width - img.shape[1]
            padding = np.zeros((self.height, rest, 3))
            img = np.concatenate((img, padding), axis=1)
            self.paddings.append(float(rest))
            return img
