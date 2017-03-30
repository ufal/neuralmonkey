from typing import Callable, IO, Iterable, List, NamedTuple

import io
import subprocess
import sys

import numpy as np

from scipy.io import wavfile

from neuralmonkey.readers.index_reader import index_reader


# pylint: disable=invalid-name
Audio = NamedTuple("Audio", [('rate', int), ('data', np.ndarray)])


def audio_reader(prefix: str="",
                 audio_format: str="wav") -> Callable:
    """Get a reader of audio files loading them from a list of pahts.

    Args:
        prefix: Prefix of the paths to the audio files.

    Returns:
        The reader function that takes a list of audio file paths (relative to
        provided prefix) and returns a list of numpy arrays.
    """

    if audio_format == "wav":
        load_file = _load_wav
    elif audio_format == "sph":
        load_file = _load_sph
    else:
        raise ValueError(
            "Unsupported audio format: {}".format(audio_format))

    def load(list_files: List[str]) -> Iterable[Audio]:
        read_list = index_reader(prefix, encoding='binary')

        for audio_file in read_list(list_files):
            yield load_file(audio_file)

    return load


def _load_wav(audio_file: IO[bytes]) -> Audio:
    """Read a WAV file."""
    return Audio(*wavfile.read(audio_file))


def _load_sph(audio_file: IO[bytes]) -> Audio:
    """Read a NIST Sphere audio file using sox."""
    process = subprocess.Popen(['sox', '-t', 'sph', '-', '-t', 'wav', '-'],
                               stdin=audio_file,
                               stdout=subprocess.PIPE,
                               stderr=sys.stderr)
    data = io.BytesIO(process.stdout.read())

    error_code = process.wait()
    if error_code != 0:
        raise RuntimeError("SoX exited with error code {} when "
                           "processing {}".format(error_code, audio_file.name))

    return Audio(*wavfile.read(data))
