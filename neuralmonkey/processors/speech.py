from typing import Callable

import numpy as np
from python_speech_features import mfcc, fbank, logfbank, ssc, delta
import tensorflow as tf


# pylint: disable=invalid-name
def SpeechFeaturesPreprocessor(feature_type: str = "mfcc",
                               delta_order: int = 0,
                               delta_window: int = 2,
                               **kwargs) -> Callable[[tf.Tensor, tf.Tensor],
                                                     tf.Tensor]:
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
            "Unknown speech feature type '{}'".format(feature_type))

    def py_preprocess(rate: int, data: np.ndarray) -> np.ndarray:
        features = [FEATURE_TYPES[feature_type](
            data, samplerate=rate, **kwargs)]

        for _ in range(delta_order):
            features.append(delta(features[-1], delta_window))

        return np.concatenate(features, axis=1).astype(np.float32)

    def preprocess(rate: tf.Tensor, data: tf.Tensor) -> tf.Tensor:

        result = tf.py_func(py_preprocess, [rate, data], tf.float32)
        result.set_shape([None, None])
        return result

    return preprocess


def _fbank(*args, **kwargs) -> np.ndarray:
    feat, _ = fbank(*args, **kwargs)
    return feat


FEATURE_TYPES = {"mfcc": mfcc,
                 "fbank": _fbank,
                 "logfbank": logfbank,
                 "ssc": ssc}
