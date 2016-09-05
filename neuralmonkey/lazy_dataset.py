"""THis module implements a dataset class that unlike dataset.Dataset does not
load the data into memory, but loads gradually from a file.
"""
# tests: mypy

import gzip
import pickle as pickle
import magic

from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.readers.plain_text_reader import PlainTextFileReader

class LazyDataset(Dataset):
    """Implements the lazy dataset by overloading the create_serie method that
    return an infinitely looping generator instead of a list.
    """

    def _check_series_lengths(self):
        """Cannot check series lengths in lazy dataset."""
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


    def __len__(self):
        raise Exception("Lazy dataset does not know its size")


    def shuffle(self):
        """Does nothing, not in-memory shuffle is impossible."""
        pass
