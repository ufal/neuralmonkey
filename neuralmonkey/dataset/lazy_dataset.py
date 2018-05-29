"""Lazy dataset which does not load the whole data into memory."""
import os
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from typeguard import check_argument_types
from neuralmonkey.dataset.dataset import Dataset

# pylint: disable=invalid-name
Reader = Callable[[List[str]], Any]
# pylint: enable=invalid-name


class LazyDataset(Dataset):
    """Implements the lazy dataset.

    The main difference between this implementation and the default one is
    that the contents of the file are not fully loaded to the memory.
    Instead, everytime the function ``get_series`` is called, a new file handle
    is created and a generator which yields lines from the file is returned.
    """

    def __init__(self, name: str,
                 series_paths_and_readers: Dict[str, Tuple[List[str], Reader]],
                 series_outputs: Dict[str, str],
                 preprocessors: List[Tuple[str, str, Callable]] = None
                ) -> None:
        """Create a new instance of the lazy dataset.

        Arguments:
            name: The name of the dataset series_paths_and_readers: The mapping
            of series name to its file series_outputs: Dictionary mapping
            series names to their output file preprocess: The preprocessor to
            apply to the read lines
        """
        check_argument_types()

        parent_series = {s: [] for s in series_paths_and_readers} \
            # type: Dict[str, List]
        if preprocessors:
            parent_series.update({s[1]: [] for s in preprocessors})
        Dataset.__init__(self, name, parent_series, series_outputs)

        self.series_paths_and_readers = series_paths_and_readers

        for series_name, (paths, _) in series_paths_and_readers.items():
            for path in paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        "File not found. Series: {}, Path: {}"
                        .format(series_name, path))

        self.preprocess_series = {} \
            # type: Dict[str, Tuple[Optional[str], Callable]]
        if preprocessors is not None:
            for src_id, tgt_id, func in preprocessors:
                if src_id == tgt_id:
                    raise Exception(
                        "Attempt to rewrite series '{}'".format(src_id))
                if src_id not in series_paths_and_readers:
                    raise Exception(
                        ("The source series ({}) of the '{}' preprocessor "
                         "is not defined in the dataset.").format(
                             src_id, str(func)))
                self.preprocess_series[tgt_id] = (src_id, func)

    def has_series(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return (name in self.series_paths_and_readers
                or name in self.preprocess_series)

    def maybe_get_series(self, name: str) -> Optional[Iterable]:
        """Get the data series with a given name or None if it does not exist.

        This function opens a new file handle and returns a generator which
        yields preprocessed lines from the file.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series or None if it does not exist.
        """
        if not self.has_series(name):
            return None

        if name in self.series_paths_and_readers:
            paths, reader = self.series_paths_and_readers[name]
            return reader(paths)

        assert name in self.preprocess_series
        src_id, func = self.preprocess_series[name]
        if src_id is None:
            return func(self)

        src_series = self.maybe_get_series(src_id)

        return None if src_series is None else (
            func(item) for item in src_series)

    def get_series(self, name: str) -> Iterable:
        """Get the data series with a given name.

        This function opens a new file handle and returns a generator which
        yields preprocessed lines from the file.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series.

        Raises:
            KeyError if the series does not exist.
        """
        if name in self.series_paths_and_readers:
            paths, reader = self.series_paths_and_readers[name]
            return reader(paths)
        elif name in self.preprocess_series:
            src_id, func = self.preprocess_series[name]
            if src_id is None:
                return func(self)

            src_series = self.get_series(src_id)
            return (func(item) for item in src_series)
        else:
            raise KeyError("Series '{}' is not in the dataset.".format(name))

    def shuffle(self) -> None:
        """Do nothing, not in-memory shuffle is impossible.

        TODO: this is related to the ``__len__`` method.
        """
        pass

    @property
    def series_ids(self) -> Iterable[str]:
        return (list(self.series_paths_and_readers.keys())
                + list(self.preprocess_series.keys()))

    def add_series(self, name: str, series: Iterable[Any]) -> None:
        raise NotImplementedError(
            "Lazy dataset does not support adding series.")

    def subset(self, start: int, length: int) -> Dataset:
        subset_name = "{}.{}.{}".format(self.name, start, length)
        subset_outputs = {k: "{}.{:010}".format(v, start)
                          for k, v in self.series_outputs.items()}

        # TODO make this more efficient with large datasets
        subset_series = {
            s_id: list(islice(self.get_series(s_id), start, start + length))
            for s_id in self.series_ids}

        return Dataset(subset_name, subset_series, subset_outputs)
