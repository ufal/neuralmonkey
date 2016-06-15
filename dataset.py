""" Implementation of the dataset class. """

import codecs
import random
from itertools import izip

import magic
import numpy as np

from utils import log

from readers.plain_text_reader import PlainTextFileReader

class Dataset(object):
    """ This class serves as collection for data series for particular
    encoders and decoders in the model. If it is not provided a parent
    dataset, it also manages the vocabularies inferred from the data.

    A data serie is either a list of strings or a numpy array.

    Attributes:

        series: Dictionary from the series name to the actual data.

        series_outputs: Output files for target series.

        random_seed: Random seed used for shuffling.

    """

    def __init__(self, **kwargs):
        """ Creates a dataset from the provided arguments. Paths to the data are
        provided in a form of dictionary.

        Only textual datasets for which the language was provided a vocabulary
        can be generated.

        Args:

            kwargs: Arguments are treated as a dictionary. Paths to the data
                series are specified here. Series identifiers should not contain
                underscores. You can specify a language for the serie by adding
                a preprocess method you want to apply on the textual data by
                naming the function as <identifier>_preprocess=function
                OR the preprocessor can be specified globally

                output file path <identifier>_out
        """

        self.preprocessor = kwargs.get('preprocessor', lambda x: x)
        self.random_seed = kwargs.get('random_seed', None)

        series_paths = self._get_series_paths(kwargs)

        if len(series_paths) > 0:
            log("Initializing dataset with: {}".format(", ".join(series_paths)))
            self._series = {s: self.create_serie(series_paths[s])
                            for s in series_paths}
            self._check_series_lengths()
            self.name = kwargs.get('name', self._get_name_from_paths(series_paths))


        # TODO make the code nicer
        self.series_outputs = {key[2:-4]: value
                               for key, value in kwargs.iteritems()
                               if key.endswith('_out') and key.startswith('s_')}


    def _get_series_paths(self, kwargs):
        # anything that is not a serie must have _
        # keys = [k for k in kwargs.keys() if k.find('_') == -1]
        # names = keys

        # all series start with s_
        keys = [k for k in kwargs.keys() if k.startswith('s_')]
        names = [k[2:] for k in keys]

        return {name : kwargs[key] for name, key in zip(names, keys)}


    def _get_name_from_paths(self, series_paths):
        name = "dataset"
        for s, path in series_paths.iteritems():
            name += "-{}".format(path)

        return name


    def _check_series_lengths(self):
        lengths = [len(v) for v in self._series.values()
                    if isinstance(v, list) or isinstance(v, np.ndarray)]

        if len(set(lengths)) > 1:
            err_str = ["{}: {}".format(s, len(self._series[s]))
                       for s in self._series]
            raise Exception("Lengths of data series must be equal. Instead: {}"
                            .format(", ".join(err_str)))


    def create_serie(self, path):
        """ Loads a data serie from a file """
        log("Loading {}".format(path))
        file_type = magic.from_file(path, mime=True)

        if file_type.startswith('text/'):
            reader = PlainTextFileReader(path)
            return list([self.preprocessor(line) for line in reader.read()])

        elif file_type == 'application/octet-stream':
            return np.load(path)
        else:
            raise Exception("\"{}\" has Unsupported data type: {}"
                            .format(path, file_type))


    def __len__(self):
        if not self._series.values():
            return 0
        else:
            return len(self._series.values()[0])

    def has_series(self, name):
        return name in self._series

    def get_series(self, name, allow_none=False):
        if allow_none:
            return self._series.get(name)
        else:
            return self._series[name]

    def shuffle(self):
        # type: None -> None
        """ Shuffles the dataset randomly """

        keys = self._series.keys()
        zipped = zip(*[self._series[k] for k in keys])
        random.shuffle(zipped)
        for key, serie in zip(keys, zip(*zipped)):
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
        keys = self._series.keys()
        batched_series = [self.batch_serie(key, batch_size) for key in keys]

        batch_index = 0
        for next_batches in izip(*batched_series):
            batch_dict = {key:data for key, data in zip(keys, next_batches)}
            dataset = Dataset(**{})
            #pylint: disable=protected-access
            dataset._series = batch_dict
            dataset.name = self.name + "-batch-{}".format(batch_index)
            batch_index += 1
            yield dataset
