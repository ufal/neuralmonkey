""" Implementation of the dataset class. """
# tests: lint, mypy
import random
import re
import collections

from typing import Callable, Dict, Type #pylint: disable=unused-import

import numpy as np
import magic

from neuralmonkey.logging import log
from neuralmonkey.readers.plain_text_reader import PlainTextFileReader

SERIES_SOURCE = re.compile("s_([^_]*)$")
SERIES_OUTPUT = re.compile("s_(.*)_out")

def load_dataset_from_files(name: str=None, lazy: bool=False,
                            preprocessor: Callable[[str], str]=lambda x: x,
                            **kwargs: str) -> 'Dataset':
    """Load a dataset from the files specified by the provided arguments.
    Paths to the data are provided in a form of dictionary.

    Keyword arguments:
        name: The name of the dataset to use. If None (default), the name will
              be inferred from the file names.
        lazy: Boolean flag specifying whether to use lazy loading (useful for
              large files). Note that the lazy dataset cannot be shuffled.
              Defaults to False.
        preprocessor: A callable used for preprocessing of the input sentences.
        kwargs: Dataset keyword argument specs. These parameters should begin
                with 's_' prefix and may end with '_out' suffix.
                For example, a data series 'source' which specify the source
                sentences should be initialized with the 's_source' parameter,
                which specifies the path to the source file.
                If the decoder generate data of the 'target' series, the output
                file should be initialized with the 's_target_out' parameter.
                Series identifiers should not contain underscores.

    Returns:
        The newly created dataset.

    Raises:
        Exception when no input files are provided.
    """
    series_paths = _get_series_paths(kwargs)
    series_outputs = _get_series_outputs(kwargs)

    if len(series_paths) == 0:
        raise Exception("No input files are provided.")

    log("Initializing dataset with: {}".format(", ".join(series_paths)))

    clazz = Dataset # type: Type[Dataset]
    if lazy:
        clazz = LazyDataset

    series = {s: clazz.create_series(series_paths[s], preprocessor)
              for s in series_paths}

    if name is None:
        name = _get_name_from_paths(series_paths)

    dataset = clazz(name, series, series_outputs)

    if not lazy:
        log("Dataset length: {}".format(len(dataset)))

    return dataset


def _get_name_from_paths(series_paths: Dict[str, str]) -> str:
    """Construct name for a dataset using the paths to its files.

    Arguments:
        series_paths: A dictionary which maps serie names to the paths
                      of their input files.

    Returns:
        The name for the dataset.
    """
    name = "dataset"
    for _, path in series_paths.items():
        name += "-{}".format(path)
    return name


def _get_series_paths(kwargs: Dict[str, str]) -> Dict[str, str]:
    """Get paths to files that contain data from the dataset keyword
    argument specs.

    Input file for a serie named 'xxx' is specified by parameter 's_xxx'

    Arguments:
        kwargs: A dictionary containing the dataset keyword argument specs.

    Returns:
        A dictionary which maps serie names to the paths of their input files.
    """
    keys = [k for k in list(kwargs.keys()) if SERIES_SOURCE.match(k)]
    names = [SERIES_SOURCE.match(k).group(1) for k in keys]

    return {name : kwargs[key] for name, key in zip(names, keys)}


def _get_series_outputs(kwargs: Dict[str, str]) -> Dict[str, str]:
    """Get paths to series outputs from the dataset keyword argument specs.
    Output file for a series named 'xxx' is specified by parameter 's_xxx_out'

    Arguments:
        kwargs: A dictionary containing the dataset keyword argument specs.

    Returns:
        A dictionary which maps serie names to the paths for their output files.
    """
    return {SERIES_OUTPUT.match(key).group(1): value
            for key, value in kwargs.items() if SERIES_OUTPUT.match(key)}


class Dataset(collections.Sized):
    """ This class serves as collection for data series for particular
    encoders and decoders in the model. If it is not provided a parent
    dataset, it also manages the vocabularies inferred from the data.

    A data series is either a list of strings or a numpy array.
    """

    def __init__(self, name, series, series_outputs, random_seed=None):
        """Creates a dataset from the provided already preprocessed
        series of data.

        Arguments:
            series: Dictionary from the series name to the actual data.
            series_outputs: Output files for target series.
            random_seed: Random seed used for shuffling.
        """

        self.name = name
        self._series = series
        self.series_outputs = series_outputs
        self.random_seed = random_seed

        self._check_series_lengths()

    def _check_series_lengths(self):
        lengths = [len(v) for v in list(self._series.values())
                   if isinstance(v, list) or isinstance(v, np.ndarray)]

        if len(set(lengths)) > 1:
            err_str = ["{}: {}".format(s, len(self._series[s]))
                       for s in self._series]
            raise Exception("Lengths of data series must be equal. Instead: {}"
                            .format(", ".join(err_str)))


    @staticmethod
    def create_series(path, preprocess=lambda x: x):
        """ Loads a data serie from a file """
        log("Loading {}".format(path))
        file_type = magic.from_file(path, mime=True)

        if file_type.startswith('text/'):
            reader = PlainTextFileReader(path)
            return list([preprocess(line) for line in reader.read()])

        elif file_type == 'application/octet-stream':
            return np.load(path)
        else:
            raise Exception("\"{}\" has Unsupported data type: {}"
                            .format(path, file_type))


    def __len__(self):
        # type: () -> int
        if not list(self._series.values()):
            return 0
        else:
            return len(list(self._series.values())[0])

    def has_series(self, name):
        # type: (str) -> bool
        return name in self._series

    def get_series(self, name, allow_none=False):
        if allow_none:
            return self._series.get(name)
        else:
            return self._series[name]

    def shuffle(self):
        # type: () -> None
        """ Shuffles the dataset randomly """

        keys = list(self._series.keys())
        zipped = list(zip(*[self._series[k] for k in keys]))
        random.shuffle(zipped)
        for key, serie in zip(keys, list(zip(*zipped))):
            self._series[key] = serie

    def batch_serie(self, serie_name, batch_size):
        """ Splits a data serie into batches """
        buf = []
        for item in self.get_series(serie_name):
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def batch_dataset(self, batch_size):
        """ Splits the dataset into a list of batched datasets. """
        keys = list(self._series.keys())
        batched_series = [self.batch_serie(key, batch_size) for key in keys]

        batch_index = 0
        for next_batches in zip(*batched_series):
            batch_dict = {key:data for key, data in zip(keys, next_batches)}
            dataset = Dataset(self.name + "-batch-{}".format(batch_index), batch_dict, {})
            batch_index += 1
            yield dataset



class LazyDataset(Dataset):
    """Implements the lazy dataset by overloading the create_serie method that
    return an infinitely looping generator instead of a list.
    """

    def _check_series_lengths(self):
        """Cannot check series lengths in lazy dataset."""
        pass


    def __len__(self):
        raise Exception("Lazy dataset does not know its size")


    def shuffle(self):
        """Does nothing, not in-memory shuffle is impossible."""
        pass


    @staticmethod
    def create_series(path, preprocess=lambda x: x):
        """ Loads a data serie from a file

        Arguments:
            path: The path to the file.
            preprocess: Function to apply to each line of the file
        """
        log("Lazy creation of a data serie from file {}".format(path))

        file_type = magic.from_file(path, mime=True)

        if file_type.startswith("text/"):
            reader = PlainTextFileReader(path)
            for line in reader.read():
                yield preprocess(line)
        else:
            raise Exception("Unsupported data type for lazy dataset:"
                            " File {}, type {}".format(path, file_type))
