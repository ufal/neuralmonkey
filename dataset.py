""" Iplementation of the dataset class. """

import codecs
import random
import magic
import numpy as np

from utils import log

class Dataset(object):
    """
    This class serves as collection for data series for particular
    encoders and decoders in the model. If it is not provided a parent
    dataset, it also manages the vocabularies inferred from the data.

    A data serie is either list of strings or a numpy array.

    Attributes:

        series: Dictionary from the series name to the actual data.

        series_outputs: Output files for target series.

        random_seed: Random seed used for shuffling.

    """

    def __init__(self, **args):
        """

        Creates a dataset from the provided arguments. Path to the data are
        provided in a form dictionary.

        Only textual datasets from the textual datasets for which the language
        was provided a vocabulary can be generated.

        Args:

            args: Arguements treated as a dictionary. Paths to the data series
                are specified here. Series identifiers should not contain
                underscore. You can scecify a language fo the serie by adding

                a preprocess method you want to
                apply on the textual data by naming the function as
                <identifier>_preprocess=function

                output file path <identifier>_out

        """

        if 'name' in args:
            self.name = args['name']
        else:
            self.name = "dataset"

        self.original_args = args
        series_names = [k for k in args.keys() if k.find('_') == -1]
        if args:
            log("Initializing dataset with: {}".format(", ".join(series_names)))


        self.series = {name: self.create_serie(name, args) for name in series_names}

        if len(set([len(v) for v in self.series.values()
                    if isinstance(v, list) or isinstance(v, np.ndarray)])) > 1:
            lengths = ["{} ({}): {}".format(s, args[s], len(self.series[s])) for s in self.series]
            raise Exception("All data series should have the same length, have: {}"\
                    .format(", ".join(lengths)))

        self.series_outputs = \
                {key[:-4]: value for key, value in args.iteritems() if key.endswith('_out')}

        if 'random_seed' in args:
            self.random_seed = args['random_seed']
        else:
            self.random_seed = None

        try:
            if args:
                log("Dataset loaded, {} examples.".format(len(self)))
        except:
            pass

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
                return list([preprocess(line.rstrip()) for line in f_data])
        elif file_type == 'application/octet-stream':
            return np.load(path)
        else:
            raise Exception("\"{}\" has Unsopported data type: {}".format(path, file_type))

    def __len__(self):
        if not self.series.values():
            return 0
        else:
            return len(self.series.values()[0])

    def shuffle(self):
        # type: None -> None
        """ Shuffles the dataset randomly """

        keys = self.series.keys()
        zipped = zip(*[self.series[k] for k in keys])
        random.shuffle(zipped)
        for key, serie in zip(keys, zip(*zipped)):
            self.series[key] = serie

    def batch_serie(self, serie_name, batch_size):
        """ Splits a data serie into batches """
        buf = []
        for item in self.series[serie_name]:
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        yield buf

    def batch_dataset(self, batch_size):
        """ Splits the dataset into a list of batched datasets. """
        keys = self.series.keys()
        batched_series = [self.batch_serie(key, batch_size) for key in keys]

        # we need to avoid using zip(*[...]) because it materializes the sequence,
        # it needs to be in the explicit while-true loop
        while True:
            next_batches = [next(bs, None) for bs in batched_series]
            if None in next_batches:
                break
            batch_dict = {key:data for key, data in zip(keys, next_batches)}
            dataset = Dataset(**{})
            dataset.series = batch_dict
            yield dataset

