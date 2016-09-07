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
