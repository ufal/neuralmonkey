"""
This module contains functions that are suppoosed to be called from the
configuration file because calling the functions or the class constructors
directly would be inconvinet or impossible.
"""
#tests: lint

import re
import os

from neuralmonkey.logging import log, debug
from neuralmonkey.dataset import Dataset, LazyDataset
from neuralmonkey.vocabulary import Vocabulary

SERIES_SOURCE = re.compile("s_([^_]*)$")
SERIES_OUTPUT = re.compile("s_(.*)_out")

def dataset_from_files(**kwargs):
    """
    Creates a dataset from the provided arguments. Paths to the data are
    provided in a form of dictionary.

    Args:

        kwargs: Arguments are treated as a dictionary. Paths to the data
            series are specified here. Series identifiers should not contain
            underscores. You can specify a language for the serie by adding
            a preprocess method you want to apply on the textual data by
            naming the function as <identifier>_preprocess=function
            OR the preprocessor can be specified globally
    """
    random_seed = kwargs.get("random_seed", None)
    preprocess = kwargs.get("preprocessor", lambda x: x)
    name = kwargs.get("name", "dataset")
    lazy = kwargs.get("lazy", False)
    series = None
    series_paths = _get_series_paths(kwargs)

    debug("Series paths: {}".format(series_paths), "datasetBuild")

    clazz = LazyDataset if lazy else Dataset

    if len(series_paths) > 0:
        log("Initializing dataset with: {}".format(", ".join(series_paths)))
        series = {s: clazz.create_series(series_paths[s], preprocess)
                  for s in series_paths}
        name = kwargs.get('name', _get_name_from_paths(series_paths))

    series_outputs = {SERIES_OUTPUT.match(key).group(1): value
                      for key, value in kwargs.items()
                      if SERIES_OUTPUT.match(key)}

    dataset = clazz(name, series, series_outputs, random_seed)

    if not lazy:
        log("Dataset length: {}".format(len(dataset)))

    return dataset


def _get_series_paths(kwargs):
    # all series start with s_
    keys = [k for k in list(kwargs.keys()) if SERIES_SOURCE.match(k)]
    names = [SERIES_SOURCE.match(k).group(1) for k in keys]

    return {name : kwargs[key] for name, key in zip(names, keys)}


def _get_name_from_paths(series_paths):
    name = "dataset"
    for _, path in series_paths.items():
        name += "-{}".format(path)

    return name


def initialize_vocabulary(directory, name, datasets=None, series_ids=None,
                          max_size=None):
    """This function is supposed to initialize vocabulary when called from the
    configuration file. It first checks whether the vocabulary is already
    loaded on the provided path and if not, it tries to generate it from
    the provided dataset.

    Arguments:
        directory: Directory where the vocabulary should be stored.
        name: Name of the vocabulary which is also the name of the file
              it is stored it.
        datasets: A a list of datasets from which the vocabulary can be
                  created.
        series_ids: A list of ids of series of the datasets that should be used
                    for producing the vocabulary.
    """
    file_name = os.path.join(directory, name + ".pickle")
    if os.path.exists(file_name):
        return Vocabulary.from_pickled(file_name)
    else:
        if datasets is None or series_ids is None or max_size is None:
            raise Exception("Vocabulary does not exist in \"{}\","+
                            "neither dataset and series_id were provided.")
        vocabulary = Vocabulary.from_datasets(datasets, series_ids, max_size)

        if not os.path.exists(directory):
            os.makedirs(directory)

        vocabulary.save_to_file(file_name)
        return vocabulary
