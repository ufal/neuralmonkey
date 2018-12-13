"""Implementation of the dataset class."""
# pylint: disable=too-many-lines
# After deleting the legacy function load_dataset_from_files, this file becomes
# short again.
import glob
import os
import random
import re

from collections import deque
from itertools import islice
from typing import (
    Any, TypeVar, Iterator, Callable, Optional, Dict, Union, List, Tuple, cast)

from typeguard import check_argument_types
from neuralmonkey.config.parsing import get_first_match
from neuralmonkey.logging import debug, log, warn
from neuralmonkey.readers.plain_text_reader import UtfPlainTextReader
from neuralmonkey.util.match_type import match_type
from neuralmonkey.writers.auto import AutoWriter
from neuralmonkey.writers.plain_text_writer import Writer

# pylint: disable=invalid-name
DataType = TypeVar("DataType")
DataSeries = Iterator[DataType]
DataExample = Dict[str, DataType]

# Reader: function that gets list of files and yields data
Reader = Callable[[List[str]], Any]

DatasetPreprocess = Callable[["Dataset"], DataSeries]
DatasetPostprocess = Callable[
    ["Dataset", Dict[str, DataSeries]], DataSeries]

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
        reader = UtfPlainTextReader
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
            reader = UtfPlainTextReader

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
         buffer_size: int = None,
         shuffled: bool = False) -> "Dataset":
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

    iterators = {}  # type: Dict[str, Callable[[], DataSeries]]

    prep_sl = {}  # type: Dict[str, Tuple[Callable, str]]
    prep_dl = {}  # type: Dict[str, DatasetPreprocess]

    def _make_iterator(reader, files):
        def itergen():
            return reader(files)
        return itergen

    def _make_sl_iterator(src, prep):
        def itergen():
            return (prep(item) for item in iterators[src]())
        return itergen

    def _make_dl_iterator(func):
        def itergen():
            return func(iterators)
        return itergen

    # First, prepare iterators for series using file readers
    for s_name, source_spec in zip(series, data):
        if match_type(source_spec, ReaderDef):  # type: ignore
            files, reader = _normalize_readerdef(cast(ReaderDef, source_spec))
            for path in files:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        "File not found. Series: {}, Path: {}"
                        .format(s_name, path))

            iterators[s_name] = _make_iterator(reader, files)

        elif match_type(source_spec, Tuple[Callable, str]):
            prep_sl[s_name] = cast(Tuple[Callable, str], source_spec)

        else:
            assert match_type(source_spec, DatasetPreprocess)  # type: ignore
            prep_dl[s_name] = cast(DatasetPreprocess, source_spec)

    # Second, prepare series-level preprocessors.
    # Note that series-level preprocessors cannot be stacked on the dataset
    # specification level.
    for s_name, (preprocessor, source) in prep_sl.items():
        if source not in iterators:
            raise ValueError(
                "Source series for series-level preprocessor nonexistent: "
                "Preprocessed series '{}', source series '{}'")
        iterators[s_name] = _make_sl_iterator(source, preprocessor)

    # Finally, dataset-level preprocessors.
    for s_name, func in prep_dl.items():
        iterators[s_name] = _make_dl_iterator(func)

    output_dict = None
    if outputs is not None:
        output_dict = {s_name: (path, writer)
                       for s_name, path, writer
                       in [_normalize_outputspec(out) for out in outputs]}

    if buffer_size is not None:
        return Dataset(name, iterators, batching, output_dict,
                       (buffer_size // 2, buffer_size), shuffled)

    return Dataset(name, iterators, batching, output_dict, None, shuffled)
# pylint: enable=too-many-locals,too-many-branches


class Dataset:
    """Buffered and batched dataset.

    This class serves as collection of data series for particular encoders and
    decoders in the model.

    Dataset has a number of data series, which are sequences of data (of any
    type) that should have the same length. The sequences are loaded in a
    buffer and can be loaded lazily.

    Using the `batches` method, dataset yields batches, through which the data
    are accessed by the model.
    """

    def __init__(self,
                 name: str,
                 iterators: Dict[str, Callable[[], Iterator]],
                 batching: BatchingScheme,
                 outputs: Dict[str, Tuple[str, Writer]] = None,
                 buffer_size: Tuple[int, int] = None,
                 shuffled: bool = False) -> None:
        """Construct a new instance of the dataset class.

        Do not call this method from the configuration directly. Instead, use
        the `from_files` function of this module.

        The dataset iterators are provided through factory functions, which
        return the opened iterators when called with no arguments.

        Arguments:
            name: The name for the dataset.
            iterators: A series-iterator generator mapping.
            lazy: If False, load the data from iterators to a list and store
                the list in memory.
            buffer_size: Use this tuple as a minimum and maximum buffer size
                for pre-loading data. This should be (a few times) larger than
                the batch size used for mini-batching. When the buffer size
                gets under the lower threshold, it is refilled with the new
                data and optionally reshuffled. If the buffer size is `None`,
                all data is loaded into memory.
            shuffled: Whether to shuffle the buffer during batching.
        """
        self.name = name
        self.iterators = iterators
        self.batching = batching
        self.outputs = outputs

        if buffer_size is not None:
            self.lazy = True
            self.buffer_min_size, self.buffer_size = buffer_size
        else:
            self.lazy = False

        self.shuffled = shuffled
        self.length = None

        if not self.lazy:
            # Load the data from iterators to memory and point new iterators
            # to these structures. (This prevents multiple loads from disk.)
            data = {s_name: list(it())
                    for s_name, it in self.iterators.items()}

            # Check whether all loaded series have the same length
            length_dict = {
                s_name: len(s_data) for s_name, s_data in data.items()}
            if len(set(length_dict.values())) > 1:
                raise ValueError("Lengths of data series do not match: {}"
                                 .format(str(length_dict)))

            self.length = next(iter(length_dict.values()))
            self.iterators = {
                s_name: lambda n=s_name: iter(data[n])  # type: ignore
                for s_name in self.iterators}

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.

        Raises:
            `NotImplementedError` when the dataset is lazy.
        """
        if self.lazy:
            raise NotImplementedError("Querying the len of a lazy dataset.")
        assert self.length is not None
        return self.length

    def __contains__(self, name: str) -> bool:
        """Check if the dataset contains a series of a given name.

        Arguments:
            name: Series name

        Returns:
            True if the dataset contains the series, False otherwise.
        """
        return name in self.iterators

    @property
    def series(self) -> List[str]:
        return list(sorted(self.iterators.keys()))

    def get_series(self, name: str) -> Iterator:
        """Get the data series with a given name.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            A freshly initialized iterator over the data series.

        Raises:
            KeyError if the series does not exists.
        """
        return self.iterators[name]()

    def maybe_get_series(self, name: str) -> Optional[Iterator]:
        """Get the data series with a given name, if it exists.

        Arguments:
            name: The name of the series to fetch.

        Returns:
            The data series or None if it does not exist.
        """
        if name in self.iterators:
            return self.get_series(name)
        return None

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def batches(self) -> Iterator["Dataset"]:
        """Split the dataset into batches.

        Returns:
            Generator yielding the batches.
        """
        if self.batching.batch_size is not None:
            max_bs = self.batching.batch_size
        else:
            assert self.batching.bucket_batch_sizes is not None
            max_bs = max(self.batching.bucket_batch_sizes)

        if self.lazy and self.buffer_min_size < max_bs:
            warn("Minimum buffer size ({}) lower than batch size ({}). "
                 "It is recommended to use large buffer size."
                 .format(self.buffer_min_size, max_bs))

        # Initialize iterators
        iterators = {s: it() for s, it in self.iterators.items()}

        # Create iterator over instances
        zipped_iterator = (
            dict(zip(iterators, row)) for row in zip(*iterators.values()))

        # Fill the buffer with initial values, shuffle optionally
        if self.lazy:
            # pylint: disable=stop-iteration-return
            # This is pylint issue https://github.com/PyCQA/pylint/issues/2158
            lbuf = list(next(zipped_iterator) for _ in range(self.buffer_size))
            # pylint: enable=stop-iteration-return
        else:
            lbuf = list(zipped_iterator)
        if self.shuffled:
            random.shuffle(lbuf)
        buf = deque(lbuf)

        def _make_datagen(rows, key):
            def itergen():
                return (row[key] for row in rows)
            return itergen

        # Iterate over the rest of the data until buffer is empty
        batch_index = 0
        buckets = [[]]  # type: List[List[DataExample]]

        if self.batching.bucket_boundaries is not None:
            buckets += [[] for _ in self.batching.bucket_boundaries]

        while buf:
            row = buf.popleft()

            if self.batching.bucket_boundaries is None:
                bucket_id = 0
            else:
                # TODO: use only specific series to determine the bucket number
                length = max(len(row[key]) for key in row)

                bucket_id = -1
                for b_id, limit in enumerate(self.batching.bucket_boundaries):
                    fits_in = length <= limit
                    tighter_fit = (
                        bucket_id == -1
                        or limit < self.batching.bucket_boundaries[
                            bucket_id])

                    if fits_in and tighter_fit:
                        bucket_id = b_id

            buckets[bucket_id].append(row)

            if self.batching.bucket_batch_sizes is None:
                assert self.batching.batch_size is not None
                is_full = len(buckets[bucket_id]) >= self.batching.batch_size
            else:
                is_full = (len(buckets[bucket_id])
                           >= self.batching.bucket_batch_sizes[bucket_id])

            if is_full:
                # Create the batch
                name = "{}.batch.{}".format(self.name, batch_index)
                data = {key: _make_datagen(buckets[bucket_id], key)
                        for key in buckets[bucket_id][0]}

                yield Dataset(
                    name=name, iterators=data, batching=self.batching)
                batch_index += 1
                buckets[bucket_id] = []

            # If lazy, refill buffer & shuffle if needed
            # Otherwise, all of the data is already loaded in the buffer.
            if self.lazy and len(buf) < self.buffer_min_size:
                # In case buffer_size is lower than batch_size
                to_add = self.buffer_size - len(buf)

                for _, item in zip(range(to_add), zipped_iterator):
                    buf.append(item)

                if self.shuffled:
                    lbuf = list(buf)
                    random.shuffle(lbuf)
                    buf = deque(lbuf)

        if not self.batching.drop_remainder:
            for bucket in buckets:
                if bucket:
                    name = "{}.batch.{}".format(self.name, batch_index)
                    data = {key: _make_datagen(bucket, key)
                            for key in bucket[0]}

                    yield Dataset(
                        name=name, iterators=data, batching=self.batching)
                    batch_index += 1
    # pylint: enable=too-many-locals,too-many-branches

    def subset(self, start: int, length: int) -> "Dataset":
        """Create a subset of the dataset.

        The sub-dataset will inherit the laziness and buffer size and shuffling
        from the parent dataset.

        Arguments:
            start: Index of the first data instance in the dataset.
            length: Number of instances to include in the subset.

        Returns:
            A subset `Dataset` object.
        """
        name = "{}.{}.{}".format(self.name, start, length)

        outputs = None
        if self.outputs is not None:
            outputs = {key: ("{}.{:010}".format(path, start), writer)
                       for key, (path, writer) in self.outputs.items()}

        slices = {s_id: lambda s=s_id: islice(self.get_series(s),
                                              start, start + length)
                  for s_id in self.iterators}

        # Here, the type: ignore is because of the tied argument to the lambda
        # function above, which made it Callable[[Any], ...] instead of just
        # Callable[[], ...].
        return Dataset(  # type: ignore
            name=name,
            iterators=slices,
            batching=self.batching,
            outputs=outputs,
            buffer_size=self.buffer_size,
            shuffled=self.shuffled)
