from typing import Callable, Tuple

import numpy as np
import python_speech_features


# pylint: disable=invalid-name
def MFCCPreprocessor(*args, **kwargs) -> Callable:
    """A wrapper for python_speech_features.mfcc."""

    def preprocess(data: Tuple[int, np.ndarray]) -> np.ndarray:
        rate, audio = data
        return python_speech_features.mfcc(audio, samplerate=rate,
                                           *args, **kwargs)

    return preprocess
