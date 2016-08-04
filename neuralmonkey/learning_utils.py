# tests: mypy

import os
import codecs
import re
import numpy as np
import tensorflow as tf
from termcolor import colored

from neuralmonkey.logging import log, log_print

try:
    #pylint: disable=unused-import,bare-except,invalid-name,import-error,no-member
    from typing import Dict, List, Union, Tuple
    from decoder import Decoder
    Hypothesis = Tuple[float, List[int]]
    Feed_dict = Dict[tf.Tensor, np.Array]
except:
    pass

def load_tokenized(text_file, preprocess=None):
    """
    Loads a tokenized text file a list of list of tokens.

    Args:

        text_file: An opened file.

        preprocess: A function/callable that (linguistically) preprocesses the
            sentences

    """

    if not preprocess:
        preprocess = lambda x: x

    return [preprocess(re.split(r"[ ]", l.rstrip())) for l in text_file]




    # if not postprocess:
    #     postprocess = lambda x: x

    # val_raw_tgt_sentences = val_dataset.get_series(decoder.data_id)
    # val_tgt_sentences = postprocess(val_raw_tgt_sentences)
