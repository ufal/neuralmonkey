"""Implementation of the dataset class."""
# pylint: disable=too-many-lines
# TODO(tf-data) Shorten!
import glob
import os
import re

from typing import Callable, Dict, Union, List, Tuple, cast

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types
from neuralmonkey.config.parsing import get_first_match
from neuralmonkey.logging import debug, log
from neuralmonkey.readers.plain_text_reader import tokenized_text_reader
from neuralmonkey.util.match_type import match_type
from neuralmonkey.vocabulary import PAD_TOKEN
from neuralmonkey.writers.auto import AutoWriter
from neuralmonkey.writers.plain_text_writer import Writer


# pylint: disable=invalid-name
# Reader: function that gets list of files and returns a dataset
Reader = Callable[[List[str]], tf.data.Dataset]

DatasetPreprocess = Callable[[Dict[str, tf.data.Dataset]], tf.data.Dataset]

FileDef = Union[str, List[str]]  # one or many files
ReaderDef = Union[FileDef, Tuple[FileDef, Reader]]  # files and optional reader

SeriesConfig = Dict[str, Union[ReaderDef, DatasetPreprocess]]

# SourceSpec: either a ReaderDef, series and preprocessor, or a dataset-level
# preprocessor
SourceSpec = Union[ReaderDef, Tuple[Callable, str], DatasetPreprocess]

# OutputSpec: Tuple of series name, path, and optionally a writer
OutputSpec = Union[Tuple[str, str], Tuple[str, str, Writer]]
# pylint: enable=invalid-name

PREPROCESSED_SERIES = re.compile("pre_([^_]*)$")
SERIES_SOURCE = re.compile("s_([^_]*)$")
SERIES_OUTPUT = re.compile("s_(.*)_out")


# pylint: disable=too-few-public-methods
# After migrating to py3.7, make this dataclass or namedtuple with defaults
class BatchingScheme:

    def __init__(self,
                 batch_size: int = None,
                 drop_remainder: bool = False,
                 bucket_boundaries: List[int] = None,
                 bucket_batch_sizes: List[int] = None,
                 ignore_series: List[str] = None) -> None:
        """Construct the baching scheme.

        Attributes:
            batch_size: Number of examples in one mini-batch.
            drop_remainder: Whether to throw out the last batch in the epoch
                if it is not complete.
            bucket_boundaries: Upper length boundaries of buckets.
            bucket_batch_sizes:  Batch size per bucket. Lenght should be
                `len(bucket_boundaries) + 1`
            ignore_series: Series to ignore during bucketing.
        """
        check_argument_types()

        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.bucket_boundaries = bucket_boundaries
        self.bucket_batch_sizes = bucket_batch_sizes

        self.ignore_series = []  # type: List[str]
        if ignore_series is not None:
            self.ignore_series = ignore_series

        if (self.batch_size is None) == (self.bucket_boundaries is None):
            raise ValueError("You must specify either batch_size or "
                             "bucket_boundaries, not both")

        if self.bucket_boundaries is not None:
            if self.bucket_batch_sizes is None:
                raise ValueError("You must specify bucket_batch_sizes")
            if len(self.bucket_batch_sizes) != len(self.bucket_boundaries) + 1:
                raise ValueError(
                    "There should be N+1 batch sizes for N bucket boundaries")
# pylint: enable=too-few-public-methods


# The protected functions below are designed to convert the ambiguous spec
# structures to a normalized form.

def _normalize_readerdef(reader_def: ReaderDef) -> Tuple[List[str], Reader]:
    if isinstance(reader_def, tuple):
        reader = reader_def[1]
        files = _normalize_filedef(reader_def[0])
    else:
        reader = tokenized_text_reader
        files = _normalize_filedef(reader_def)

    return files, reader


def _normalize_outputspec(output_spec: OutputSpec) -> Tuple[str, str, Writer]:
    if len(output_spec) == 2:
        return output_spec[0], output_spec[1], AutoWriter
    return cast(Tuple[str, str, Writer], output_spec)


def _normalize_filedef(file_def: FileDef) -> List[str]:
    if isinstance(file_def, str):
        return _expand_patterns_flat([file_def])
    return _expand_patterns_flat(file_def)


def _expand_patterns_flat(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matched_files = sorted(glob.glob(pattern))
        if not matched_files:
            raise FileNotFoundError(
                "Pattern did not match any files: {}".format(pattern))
        paths.extend(matched_files)

    return paths


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
            reader = tokenized_text_reader

        if isinstance(patterns, str):
            patterns = [patterns]

        paths = _expand_patterns_flat(patterns)
        debug("Series '{}' has the following files: {}".format(name, paths))

        series_sources[name] = (paths, reader)

    return series_sources


def _get_series_outputs(series_config: SeriesConfig) -> List[OutputSpec]:
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
    return [(key, val, AutoWriter) for key, val in outputs.items()]


# pylint: disable=too-many-locals,too-many-branches
def load(name: str,
         series: List[str],
         data: List[SourceSpec],
         batching: BatchingScheme = None,
         outputs: List[OutputSpec] = None,
         buffer_size: int = None) -> "Dataset":
    """Create a dataset using specification from the configuration.

    The dataset provides iterators over data series. The dataset has a buffer,
    which pre-fetches a given number of the data series lazily. In case the
    dataset is not lazy (buffer size is `None`), the iterators are built on top
    of in-memory arrays. Otherwise, the iterators operate on the data sources
    directly.

    Arguments:
        name: The name of the dataset.
        series: A list of names of data series the dataset contains.
        data: The specification of the data sources for each series.
        outputs: A list of output specifications.
        buffer_size: The size of the buffer. If set, the dataset will be loaded
            lazily into the buffer (useful for large datasets). The buffer size
            specifies the number of sequences to pre-load. This is useful for
            pseudo-shuffling of large data on-the-fly. Ideally, this should be
            (much) larger than the batch size. Note that the buffer gets
            refilled each time its size is less than half the `buffer_size`.
            When refilling, the buffer gets refilled to the specified size.
    """
    check_argument_types()

    if batching is None:
        from neuralmonkey.experiment import Experiment
        log("Using default batching scheme for dataset {}.".format(name))
        # pylint: disable=no-member
        batch_size = Experiment.get_current().config.args.batch_size
        # pylint: enable=no-member
        if batch_size is None:
            raise ValueError("Argument main.batch_size is not specified, "
                             "cannot use default batching scheme.")
        batching = BatchingScheme(batch_size=batch_size)

    if not series:
        raise ValueError("No dataset series specified.")

    if not [s for s in data if match_type(s, ReaderDef)]:  # type: ignore
        raise ValueError("At least one data series should be from a file")

    if len(series) != len(data):
        raise ValueError(
            "The 'series' and 'data' lists should have the same number"
            " of elements: {} vs {}.".format(len(series), len(data)))

    if len(series) != len(set(series)):
        raise ValueError("There are duplicate series.")

    if outputs is not None:
        output_sources = [o[0] for o in outputs]
        if len(output_sources) != len(set(output_sources)):
            raise ValueError("Multiple outputs for a single series")

    log("Initializing dataset {}.".format(name))

    data_series = {}  # type: Dict[str, tf.data.Dataset]

    prep_sl = {}  # type: Dict[str, Tuple[Callable, str]]
    prep_dl = {}  # type: Dict[str, DatasetPreprocess]

    # First, prepare iterators for series using file readers
    for s_name, source_spec in zip(series, data):
        if match_type(source_spec, ReaderDef):  # type: ignore
            files, reader = _normalize_readerdef(cast(ReaderDef, source_spec))
            for path in files:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        "File not found. Series: {}, Path: {}"
                        .format(s_name, path))

            data_series[s_name] = reader(files)

        elif match_type(source_spec, Tuple[Callable, str]):
            prep_sl[s_name] = cast(Tuple[Callable, str], source_spec)

        else:
            assert match_type(source_spec, DatasetPreprocess)  # type: ignore
            prep_dl[s_name] = cast(DatasetPreprocess, source_spec)

    # Second, prepare series-level preprocessors.
    # Note that series-level preprocessors cannot be stacked on the dataset
    # specification level.
    for s_name, (preprocessor, source) in prep_sl.items():
        if source not in data_series:
            raise ValueError(
                "Source series for series-level preprocessor nonexistent: "
                "Preprocessed series '{}', source series '{}'")
        # TODO(tf-data) num_parallel_calls
        data_series[s_name] = data_series[source].map(preprocessor)

    # Finally, dataset-level preprocessors.
    for s_name, func in prep_dl.items():
        data_series[s_name] = func(data_series)

    output_dict = None
    if outputs is not None:
        output_dict = {s_name: (path, writer)
                       for s_name, path, writer
                       in [_normalize_outputspec(out) for out in outputs]}

    return Dataset(name, data_series, batching, output_dict, buffer_size)
# pylint: enable=too-many-locals,too-many-branches


class Dataset:
    """Aggregate dataset class for input series.

    Dataset has a number of data series, which are sequences of data (of any
    type) that should have the same length. The sequences are loaded lazily in
    a tf.data.Dataset instance.

    Using the `get_dataset` method, this class returns zipped tf.data.Dataset,
    through which the data are accessed by the model parts.
    """

    def __init__(self,
                 name: str,
                 data_series: Dict[str, tf.data.Dataset],
                 batching: BatchingScheme,
                 outputs: Dict[str, Tuple[str, Writer]] = None,
                 buffer_size: int = None) -> None:
        """Construct a new instance of the dataset class.

        Do not call this method from the configuration directly. Instead, use
        the `load` function of this module.

        Arguments:
            name: The name for the dataset.
            data_series: A mapping of data_ids to `tf.data.Dataset` objects.
            batching: `BatchingScheme` for batching this dataset.
            outputs: A mapping of data_ids to files and writers.
            buffer_size: Buffer size for shuffling the dataset. If None, the
                dataset will not be shuffled.
        """
        self.name = name
        self.data_series = data_series
        self.batching = batching
        self.outputs = outputs
        self.buffer_size = buffer_size
        # TODO(tf-data) Use 'cache' for nonlazy datasets

    def __contains__(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return name in self.data_series

    def get_dataset(self,
                    types: Dict[str, tf.DType],
                    shapes: Dict[str, tf.TensorShape],
                    ignore_errors: bool = False) -> tf.data.Dataset:
        """Construct the associated tf.data.Dataset object.

        Arguments:
            types: A mapping of model part data series to their types.
            shapes: A mapping of model part data series to their shapes.
            ignore_errors: If True, do not raise exception when a series is
                present in the model but missing in the dataset.

        Returns:
            tf.data.Dataset, ready to be registered in feedables.
        """
        assert types.keys() == shapes.keys()

        # TODO(tf-data) should we check if provided types and shapes match the
        # data series in this dataset?

        add_series = {}  # Dict[str, tf.data.Dataset]

        # Data series that are missing in the types but are present in dataset
        miss_feed = [s_id for s_id in self.data_series if s_id not in types]

        # Data series required by the model not present in the dataset
        miss_data = [s_id for s_id in types if s_id not in self.data_series]

        if miss_feed:
            # Some series in the dataset are not used directly by the model.
            # They can be e.g. inputs to preprocessors.
            msg = ("Unused data series (not present in model): {}".format(
                ", ".join(miss_feed)))
            log(msg)

        # We should include input series not used by the model to be able to
        # print them out at the end.
        use_series = {s_id: data for s_id, data in self.data_series.items()}

        if miss_data:
            msg = "Dataset does not include feedable series: {}".format(
                ", ".join(miss_data))
            if ignore_errors:
                log(msg)
            else:
                raise ValueError(msg)

            # If there are series missing from the dataset and ignore_errors
            # is True, we fill the series with dummy data.
            log("Filling missing series with dummy values {}"
                .format(", ".join(miss_data)))

            def get_dummy_gen(shape, val):
                def gen():
                    while True:
                        yield np.full(shape, val)
                return gen

            for s_id in miss_data:
                dtype = types[s_id]
                shape = [dim if dim is not None else 1
                         for dim in shapes[s_id].as_list()[1:]]

                if dtype == tf.string:
                    add_series[s_id] = tf.data.Dataset.from_generator(
                        get_dummy_gen(shape, ""), dtype, shape)

                elif dtype == tf.bool:
                    add_series[s_id] = tf.data.Dataset.from_generator(
                        get_dummy_gen(shape, False), dtype, shape)
                else:  # regard data as numeric
                    add_series[s_id] = tf.data.Dataset.from_generator(
                        get_dummy_gen(shape, 0), dtype, shape)

        dataset = tf.data.Dataset.zip({**use_series, **add_series})

        if self.buffer_size:
            # pylint: disable=redefined-variable-type
            dataset = dataset.shuffle(self.buffer_size)
            # pylint: enable=redefined-variable-type

        shapes = dataset.output_shapes

        # TODO(tf-data)
        # Set num_parallel_calls everywhere possible according to
        # tf_manager.num_threads

        def make_zero(t):
            """Adapted from TF dataset_ops."""
            if t.base_dtype == tf.string:
                return PAD_TOKEN
            if t.base_dtype == tf.variant:
                raise TypeError("Unable to create padding for field of type "
                                "'variant'")
            return np.zeros_like(t.as_numpy_dtype())

        padding_values = tf.contrib.framework.nest.map_structure(
            make_zero, dataset.output_types)

        if self.batching.bucket_boundaries is not None:
            assert self.batching.bucket_batch_sizes is not None

            def bucketing_hash_function(
                    data_point: Dict[str, tf.Tensor]) -> tf.Tensor:

                lengths = [tf.shape(item) for key, item in data_point.items()
                           if key not in self.batching.ignore_series]

                return tf.reduce_max(tf.concat(lengths, axis=0))

            dataset = dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    bucketing_hash_function,
                    bucket_boundaries=self.batching.bucket_boundaries,
                    bucket_batch_sizes=self.batching.bucket_batch_sizes,
                    padded_shapes=shapes,
                    padding_values=padding_values))
        else:
            assert self.batching.batch_size

            dataset = dataset.padded_batch(
                batch_size=self.batching.batch_size,
                padded_shapes=shapes,
                padding_values=padding_values,
                drop_remainder=self.batching.drop_remainder)

        dataset = dataset.prefetch(1)

        # TODO(tf-data) repeat and skip will be added in learning_utils
        return dataset

    def subset(self, start: int, length: int) -> tf.data.Dataset:
        """Create a subset of the dataset.

        Arguments:
            start: Index of the first data instance in the dataset.
            length: Number of instances to include in the subset.

        Returns:
            A subset tf.data.Dataset object
        """
        # name = "{}.{}.{}".format(self.name, start, length)

        # outputs = None
        # if self.outputs is not None:
        #     outputs = {key: ("{}.{:010}".format(path, start), writer)
        #                for key, (path, writer) in self.outputs.items()}

        # slices = {s_id: lambda s=s_id: islice(self.get_series(s),
        #                                       start, start + length)
        #           for s_id in self.iterators}

        # # Here, the type ignore is because of the tied argument to the lambda
        # # function above, which made it Callable[[Any], ...] instead of just
        # # Callable[[], ...].
        # return Dataset(  # type: ignore
        #     name=name,
        #     iterators=slices,
        #     batching=self.batching,
        #     outputs=outputs,
        #     buffer_size=self.buffer_size,
        #     shuffled=self.shuffled)

        # TODO(tf-data) HOW to assign different write_outs and such?
        raise NotImplementedError("Not implemented yet!")
