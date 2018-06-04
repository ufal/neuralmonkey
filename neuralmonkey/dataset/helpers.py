"""Helper functions for building datasets."""

import glob
import re
from typing import cast, Any, Callable, Dict, Iterable, List, Tuple, Union

from typeguard import check_argument_types

from neuralmonkey.config.parsing import get_first_match
from neuralmonkey.dataset.dataset import Dataset
from neuralmonkey.dataset.lazy_dataset import LazyDataset, Reader
from neuralmonkey.logging import log, debug
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader

# pylint: disable=invalid-name
DatasetPreprocess = Callable[["Dataset"], Iterable[Any]]
DatasetPostprocess = Callable[
    ["Dataset", Dict[str, Iterable[Any]]], Iterable[Any]]

ReaderDef = Union[
    str, List[str], Tuple[str, Reader], Tuple[List[str], Reader]]

SeriesConfig = Dict[str, Union[ReaderDef, DatasetPreprocess]]
# pylint: enable=invalid-name


PREPROCESSED_SERIES = re.compile("pre_([^_]*)$")
SERIES_SOURCE = re.compile("s_([^_]*)$")
SERIES_OUTPUT = re.compile("s_(.*)_out")


def from_files(
        name: str, lazy: bool = False,
        preprocessors: List[Tuple[str, str, Callable]] = None,
        **kwargs) -> Dataset:
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
                with 's_' prefix and may end with '_out' suffix.  For example,
                a data series 'source' which specify the source sentences
                should be initialized with the 's_source' parameter, which
                specifies the path and optinally reader of the source file. If
                runners generate data of the 'target' series, the output file
                should be initialized with the 's_target_out' parameter.
                Series identifiers should not contain underscores.
                Dataset-level preprocessors are defined with 'pre_' prefix
                followed by a new series name. In case of the pre-processed
                series, a callable taking the dataset and returning a new
                series is expected as a value.

    Returns:
        The newly created dataset.

    Raises:
        Exception when no input files are provided.
    """
    check_argument_types()

    series_paths_and_readers = _get_series_paths_and_readers(kwargs)
    series_outputs = _get_series_outputs(kwargs)

    if not series_paths_and_readers:
        raise Exception("No input files are provided.")

    log("Initializing dataset with: {}".format(
        ", ".join(series_paths_and_readers)))

    if lazy:
        dataset = LazyDataset(name, series_paths_and_readers, series_outputs,
                              preprocessors)  # type: Dataset
    else:
        series = {key: list(reader(paths))
                  for key, (paths, reader) in series_paths_and_readers.items()}

        dataset = Dataset(name, series, series_outputs, preprocessors)
        log("Dataset length: {}".format(len(dataset)))

    _preprocessed_datasets(dataset, kwargs)

    return dataset


def _preprocessed_datasets(
        dataset: Dataset,
        series_config: SeriesConfig) -> None:
    """Apply dataset-level preprocessing."""
    keys = [key for key in series_config.keys()
            if PREPROCESSED_SERIES.match(key)]

    for key in keys:
        name = get_first_match(PREPROCESSED_SERIES, key)
        preprocessor = cast(DatasetPreprocess, series_config[key])

        if isinstance(dataset, Dataset):
            new_series = list(preprocessor(dataset))
            dataset.add_series(name, new_series)
        elif isinstance(dataset, LazyDataset):
            dataset.preprocess_series[name] = (None, preprocessor)


def _get_series_paths_and_readers(
        series_config: SeriesConfig) -> Dict[str, Tuple[List[str], Reader]]:
    """Get paths to files that contain data from the dataset kwargs.

    Input file for a serie named 'xxx' is specified by parameter 's_xxx'. The
    dataset series is defined by a string with a path / list of strings with
    paths, or a tuple whose first member is a path or a list of paths and the
    second memeber is a reader function.

    The paths can contain wildcards, which will be expanded using
    :py:func:`glob.glob` in sorted order.

    Arguments:
        series_config: A dictionary containing the dataset keyword argument
            specs.

    Returns:
        A dictionary which maps serie names to the paths of their input files
        and readers..
    """
    keys = [k for k in list(series_config.keys()) if SERIES_SOURCE.match(k)]
    names = [get_first_match(SERIES_SOURCE, k) for k in keys]

    series_sources = {}
    for name, key in zip(names, keys):
        value = cast(ReaderDef, series_config[key])

        if isinstance(value, tuple):
            patterns, reader = value  # type: ignore
        else:
            patterns = value
            reader = UtfPlainTextReader

        if isinstance(patterns, str):
            patterns = [patterns]

        paths = []
        for pattern in patterns:
            matched_files = sorted(glob.glob(pattern))
            if not matched_files:
                raise FileNotFoundError(
                    "Pattern did not match any files. Series: {}, Pattern: {}"
                    .format(name, pattern))
            paths.extend(matched_files)

        debug("Series '{}' has the following files: {}".format(name, paths))

        series_sources[name] = (paths, reader)

    return series_sources


def _get_series_outputs(series_config: SeriesConfig) -> Dict[str, str]:
    """Get paths to series outputs from the dataset keyword argument specs.

    Output file for a series named 'xxx' is specified by parameter 's_xxx_out'

    Arguments:
        series_config: A dictionary containing the dataset keyword argument
           specs.

    Returns:
        A dictionary which maps serie names to the paths for their output
        files.
    """
    outputs = {}
    for key, value in series_config.items():
        matcher = SERIES_OUTPUT.match(key)
        if matcher:
            name = matcher.group(1)
            if not isinstance(value, str):
                raise ValueError(
                    "Output path for '{}' series must be a string, was {}.".
                    format(name, type(value)))
            outputs[name] = cast(str, value)
    return outputs


# pylint: disable=invalid-name
load_dataset_from_files = from_files
