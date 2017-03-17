from typing import Callable

import numpy as np
from python_speech_features import mfcc, fbank, logfbank, ssc, delta

from neuralmonkey.readers.audio_reader import Audio


FEATURE_TYPES = {f.__name__: f for f in [mfcc, fbank, logfbank, ssc]}


# pylint: disable=invalid-name
def SpeechFeaturesPreprocessor(feature_type='mfcc', delta_order=2,
                               delta_window=2, **kwargs) -> Callable:
    """Calculate speech features.

    By default, compute 13 MFCCs + delta + acceleration, i.e. 39 coefficients.

    Arguments:
        feature_type: mfcc, fbank, logfbank or ssc (default is mfcc)
        delta_order: maximum order of the delta features (default is 2)
        delta_window: window size for delta features (default is 2)
        **kwargs: keyword arguments for the appropriate function from
            python_speech_features
    """

    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            'Unknown speech feature type "{}"'.format(feature_type))

    def preprocess(data: Audio) -> np.ndarray:
        rate, audio = data

        features = [FEATURE_TYPES[feature_type](audio, samplerate=rate,
                                                **kwargs)]

        for _ in range(delta_order):
            features.append(delta(features[-1], delta_window))

        return np.concatenate(features, axis=1)

    return preprocess
