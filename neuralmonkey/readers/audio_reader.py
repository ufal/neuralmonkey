from typing import Callable, Iterable, List, NamedTuple

import io
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
import tensorflow as tf


class Audio(NamedTuple("Audio", [("rate", int), ("data", np.ndarray)])):
    """A raw audio object with its rate as metadata.

    Attribute:
        rate: The sample rate of the audio.
        data: The raw audio data.
    """


def audio_reader(
        prefix: str = "",
        audio_format: str = "wav") -> Callable[[List[str]], tf.data.Dataset]:
    gen = py_audio_reader(prefix, audio_format)

    def reader(files: List[str]) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            lambda: gen(files),
            output_types=(tf.int32, tf.float32),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))

    return reader


def py_audio_reader(prefix: str = "",
                    audio_format: str = "wav") -> Callable:
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
        for list_file in list_files:
            with open(list_file) as f_list:
                for audio_file in f_list:
                    path = os.path.join(prefix, audio_file.rstrip())
                    yield load_file(path)

    return load


def _load_wav(path: str) -> Audio:
    """Read a WAV file."""
    return Audio(*wavfile.read(path))


def _load_sph(path: str) -> Audio:
    """Read a NIST Sphere audio file using the sph2pipe utility."""
    process = subprocess.Popen(["sph2pipe", "-f", "wav", path],
                               stdout=subprocess.PIPE,
                               stderr=sys.stderr)
    data = io.BytesIO(process.stdout.read())

    error_code = process.wait()
    if error_code != 0:
        raise RuntimeError("sph2pipe exited with error code {} when "
                           "processing {}".format(error_code, path))

    return Audio(*wavfile.read(data))
