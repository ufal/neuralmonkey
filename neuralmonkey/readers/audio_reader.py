from typing import Callable, Iterable, List, Tuple

import os

import numpy as np

from scipy.io import wavfile


def audio_reader(prefix: str="",
                 audio_format: str="wav") -> Callable:
    """Get a reader of audio files loading them from a list of pahts.

    Args:
        prefix: Prefix of the paths to the audio files.

    Returns:
        The reader function that takes a list of audio file paths (relative to
        provided prefix) and returns a list of numpy arrays.
    """

    def load(list_files: List[str]) -> Iterable[Tuple[int, np.ndarray]]:
        if audio_format == "wav":
            load_file = wavfile.read
        else:
            raise ValueError(
                "Unsupported audio format: {}".format(audio_format))

        for list_file in list_files:
            with open(list_file) as f_list:
                for audio_file in f_list:
                    path = os.path.join(prefix, audio_file.rstrip())
                    yield load_file(path)

    return load
