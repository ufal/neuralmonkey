from typing import Callable

import numpy as np
from python_speech_features import mfcc, fbank, logfbank, ssc, delta

from neuralmonkey.readers.audio_reader import Audio


# pylint: disable=invalid-name
def SpeechFeaturesPreprocessor(feature_type: str='mfcc', delta_order: int=0,
                               delta_window: int=2, **kwargs) -> Callable:
    """Calculate speech features.

    First, the given type of features (e.g. MFCC) is computed using a window
    of length `winlen` and step `winstep`; for additional keyword arguments
    (specific to each feature type), see
    http://python-speech-features.readthedocs.io/. Then, delta features up to
    `delta_order` are added.

    By default, 13 MFCCs per frame are computed. To add delta and delta-delta
    features (resulting in 39 coefficients per frame), set `delta_order=2`.

    Arguments:
        feature_type: mfcc, fbank, logfbank or ssc (default is mfcc)
        delta_order: maximum order of the delta features (default is 0)
        delta_window: window size for delta features (default is 2)
        **kwargs: keyword arguments for the appropriate function from
            python_speech_features

    Returns:
        A numpy array of shape [num_frames, num_features].
    """

    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            'Unknown speech feature type "{}"'.format(feature_type))

    def preprocess(audio: Audio) -> np.ndarray:
        features = [FEATURE_TYPES[feature_type](
            audio.data, samplerate=audio.rate, **kwargs)]

        for _ in range(delta_order):
            features.append(delta(features[-1], delta_window))

        return np.concatenate(features, axis=1)

    return preprocess


def _fbank(*args, **kwargs) -> np.ndarray:
    feat, _ = fbank(*args, **kwargs)
    return feat


FEATURE_TYPES = {'mfcc': mfcc,
                 'fbank': _fbank,
                 'logfbank': logfbank,
                 'ssc': ssc}
