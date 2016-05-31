"""
This module implements a dataset class that unlike dataset.Dataset does not
load the data into memory, but loads gradually from a file.
"""

import codecs
import magic

from utils import log
from dataset import Dataset

class LazyDataset(Dataset):
    """

    Implements the lazy dataset by overloading the create_serie method that
    return an infinitely looping generator instead of a list.

    """

    def __init__(self, **args):
        super(LazyDataset, self).__init__(**args)


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
        # TODO add pickled numpy objects
        else:
            raise Exception("\"{}\" has Unsopported data type: {}".format(path, file_type))

    def __len__(self):
        raise Exception("Lazy dataset does not know its size")

    def shuffle(self):
        """
        Instead of shuffling the dataset, it restarts the file reading.
        """
        self.series = {name: self.create_serie(name, self.original_args)
                       for name in self.series.keys()}
