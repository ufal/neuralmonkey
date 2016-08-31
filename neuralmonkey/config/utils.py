"""This module contains helper functions that are suppoosed to be called from
the configuration file because calling the functions or the class constructors
directly would be inconvinent or impossible.

"""
#tests: lint

import os

from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.dataset import load_dataset_from_files

#pylint: disable=invalid-name
# for backwards compatibility
dataset_from_files = load_dataset_from_files

def vocabulary_from_file(path):
    """Loads vocabulary from a pickled file

    Arguments:
        filename: File name to load the vocabulary from
    """
    if not os.path.exists(path):
        raise Exception("Vocabulary file does not exist: {}".format(path))
    return Vocabulary.from_pickled(path)


def vocabulary_from_dataset(datasets, series_ids, max_size, save_file=None,
                            overwrite=False):
    """Loads vocabulary from a dataset with an option to save it.

    Arguments:
        datasets: A list of datasets from which to create the vocabulary
        series_ids: A list of ids of series of the datasets that should be used
                    producing the vocabulary
        max_size: The maximum size of the vocabulary
        save_file: A file to save the vocabulary to. If None (default),
                   the vocabulary will not be saved.
    """
    vocabulary = Vocabulary.from_datasets(datasets, series_ids, max_size)

    if not overwrite and os.path.exists(save_file):
        raise Exception("Cannot save the vocabulary. File exists: {}"
                        .format(save_file))

    directory = os.path.dirname(save_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    vocabulary.save_to_file(save_file)
    return vocabulary


def vocabulary_from_bpe(path):
    """Loads vocabulary from Byte-pair encoding merge list.

    NOTE: The frequencies of words in this vocabulary are not computed from
    data. Instead, they correspond to the number of times the subword units
    occurred in the BPE merge list. This means that smaller words will tend to
    have larger frequencies assigned and therefore the truncation of the
    vocabulary can be somehow performed (but not without a great deal of
    thought).

    Arguments:
        path: File name to load the vocabulary from.
    """
    if not os.path.exists(path):
        raise Exception("BPE file does not exist: {}".format(path))
    return Vocabulary.from_bpe(path)


def initialize_vocabulary(directory, name, datasets=None, series_ids=None,
                          max_size=None):
    """This function is supposed to initialize vocabulary when called from
    the configuration file. It first checks whether the vocabulary is already
    loaded on the provided path and if not, it tries to generate it from
    the provided dataset.

    Args:
        directory: Directory where the vocabulary should be stored.

        name: Name of the vocabulary which is also the name of the file
              it is stored it.

        datasets: A a list of datasets from which the vocabulary can be
                  created.

        series_ids: A list of ids of series of the datasets that should be used
                    for producing the vocabulary.

        max_size: The maximum size of the vocabulary
    """
    log("Warning! Use of deprecated initialize_vocabulary method. "
        "Did you think this through?", color="red")

    file_name = os.path.join(directory, name + ".pickle")
    if os.path.exists(file_name):
        return vocabulary_from_file(file_name)

    if datasets is None or series_ids is None or max_size is None:
        raise Exception("Vocabulary does not exist in \"{}\","+
                        "neither dataset and series_id were provided.")

    return vocabulary_from_dataset(datasets, series_ids, max_size,
                                   save_file=file_name, overwrite=False)
