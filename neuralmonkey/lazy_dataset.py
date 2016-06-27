"""
This module implements a dataset class that unlike dataset.Dataset does not
load the data into memory, but loads gradually from a file.
"""

# tests: mypy

import codecs
import gzip
import cPickle as pickle
import magic

from neuralmonkey.utils import log
from neuralmonkey.dataset import Dataset

class LazyDataset(Dataset):
    """

    Implements the lazy dataset by overloading the create_serie method that
    return an infinitely looping generator instead of a list.

    """

    def __init__(self, **args):
        super(LazyDataset, self).__init__(**args)

# TODO 'serie' is not an English word

    def create_serie(self, name, args):
        """ Loads a data serie from a file """
        path = args[name]
        log("Loading {}".format(path))
        file_type = magic.from_file(path, mime=True)

        # if the dataset has no name, generate it from files
        if 'name' not in args:
            self.name += "-"+path

        if file_type.startswith('text/'):
            if name+"_preprocess" in args:
                preprocess = args[name+"_preprocess"]
            else:
                preprocess = lambda s: s.split(" ")

            with codecs.open(path, 'r', 'utf-8') as f_data:
                for line in f_data:
                    yield preprocess(line.rstrip())
        elif file_type == 'application/gzip':
            with gzip.open(path, 'rb') as f_data:
                try:
                    while True:
                        yield pickle.load(f_data)
                except EOFError:
                    pass
        else:
            raise Exception("\"{}\" has Unsopported data type: {}".format(path, file_type))

    def get_series(self, name, allow_none=False):
        if allow_none and name not in self.original_args:
            return None
        else:
            return self.create_serie(name, self.original_args)

    def __len__(self):
        raise Exception("Lazy dataset does not know its size")

    def shuffle(self):
        """
        Does nothing, not in-memory shuffle is impossible.
        """
        pass

    def batch_dataset(self, batch_size):
        return super(LazyDataset, self).batch_dataset(batch_size)
