"""
Abstract base classes and core utilities for the BIL data API.
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property

import numpy as np
import pandas as pd
import h5py

from .utils import fetch

REMOTE_HTTPS_URL = "https://stacks.stanford.edu/v2/file/zz618yg1930/version/2/data/"


def needs_data(*names):
    """Decorator ensuring specific data files are downloaded before method execution.

    Automatically calls `get_data(*names)` on the Study object.

    Args:
        *names: Semantic names of data files required (e.g., 'lfp', 'raster').

    Returns:
        callable: A decorator function.
    """

    def inner(func):
        """Inner decorator factory."""

        @functools.wraps(func)
        def decorator(self, *args, **kwargs):
            """Wrapper that fetches required data before execution."""
            if hasattr(self, "study"):
                self.study.get_data(*names)
            elif hasattr(self, "get_data"):
                self.get_data(*names)
            else:
                raise RuntimeError("trying to decorate invalid object")

            return func(self, *args, **kwargs)

        return decorator

    return inner


def by_time_ms(study, key, span_cls):
    """Standard implementation of by_time indexing in milliseconds.

    Args:
        study (StudyMixin): The parent study object.
        key (int or slice): The time index or range in milliseconds.
        span_cls (type): The class to instantiate for the resulting span.

    Returns:
        Span: A new span object covering the requested time.

    Raises:
        IndexError: If the requested time is out of bounds.
        ValueError: If the key type is not supported.
    """
    if isinstance(key, (int, np.integer)):
        if not 0 <= key <= study.tlen:
            raise IndexError(f"key {key} out of bounds [0, {study.tlen}]")
        start = key
        stop = start + 1
    elif isinstance(key, slice):
        if isinstance(key.start, (int, np.integer)) or isinstance(
            key.stop, (int, np.integer)
        ):
            self_range = range(0, study.tlen)
            sub_range = self_range[key]
            start = sub_range.start
            stop = sub_range.stop
        else:
            # empty slice gets complete span
            start = 0
            stop = study.tlen
    else:
        raise ValueError("`by_time` only supports int and slice indexing")

    return span_cls(start, stop, study_id=study.study_id, study=study, metadata=None)


class Span(ABC):
    """Abstract base class representing a continuous time interval of data.

    A Span defines a start and stop time in milliseconds and provides
    methods for slicing, extending, and retrieving data within that interval.

    Attributes:
        start (int): Start time of the span (inclusive).
        stop (int): Stop time of the span (exclusive).
        study_id (str): Identifier for the study this span belongs to.
        study (StudyMixin): The parent Study object.
        metadata (dict, optional): Associated trial or event metadata.
    """

    def __init__(self, start, stop, study_id, study, metadata=None, **kwargs):
        """Initialize a new Span object.

        Args:
            start (int): Start time of the interval in milliseconds relative
                to the beginning of the recording session.
            stop (int): Stop time of the interval in milliseconds (exclusive).
            study_id (str): Unique identifier for the recording session.
            study (StudyMixin): Reference to the parent Study object that
                manages data for this session.
            metadata (dict or pd.Series, optional): Collection of experimental
                parameters or trial metadata associated with this time window.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        self.start = start
        self.stop = stop
        self.study_id = study_id
        self.study = study
        self.metadata = metadata
        super().__init__(**kwargs)

    def __repr__(self):
        """Return string representation of the Span."""
        return (
            f"span(study_id={self.study_id}, start={self.start}ms, stop={self.stop}ms)"
        )

    def __getitem__(self, key):
        """Index into the span to get metadata or a sub-span.

        Args:
            key (str, int, or slice): If str, returns metadata value.
                If int or slice, returns a new sub-span.

        Returns:
            The metadata value or a new Span object.

        Raises:
            ValueError: If the key type is not supported.
            KeyError: If metadata is None and a string key is requested.
        """
        # metadata access
        if isinstance(key, str):
            if self.metadata is None:
                raise KeyError(f"Metadata is not initialized for span {self}")
            return self.metadata[key]

        # sub-span
        self_range = range(self.start, self.stop)
        if isinstance(key, int):
            start = self_range[key]
            stop = start + 1
        elif isinstance(key, slice):
            # slicing backward in time
            if key.start is not None and key.start < 0:
                if key.stop is None:
                    new_sl = slice(None, None)
                else:
                    assert key.stop > key.start
                    new_sl = slice(None, key.stop + key.start)
                return self.extend(abs(key.start))[new_sl]

            sub_range = self_range[key]
            start = sub_range.start
            stop = sub_range.stop
        else:
            raise ValueError("must index with int or slice")

        return type(self)(start, stop, self.study_id, self.study, self.metadata)

    def __len__(self):
        """Return the duration of the span in milliseconds."""
        return self.stop - self.start

    @abstractmethod
    def _data(self, func, **kwargs):
        """Abstract method for internal data retrieval."""

    def around(self, t, t_before, t_after):
        """Create a new span centered around a time point within this span.

        If the requested interval exceeds the current span's boundaries, the span
        is automatically extended using the `extend` method.

        Args:
            t (int): The center time point relative to the start of this span.
            t_before (int): Duration before the time point.
            t_after (int): Duration after the time point.

        Returns:
            Span: A new span covering [t - t_before, t + t_after + 1].
        """
        start = t - t_before
        end = t + t_after + 1

        span = self
        if self.start + end > self.stop:
            span = span.extend(t_after=self.start + end - self.stop)
            end = -1
        if start < 0:
            span = span.extend(t_before=abs(start))
            if end != -1:
                end += abs(start)
            start = 0

        if end == -1:
            return span[start:]
        return span[start:end]

    def extend(self, t_before=0, t_after=0):
        """Return a span extended beyond the original bounds.

        Args:
            t_before (int): Amount to extend backward from the start.
            t_after (int): Amount to extend forward from the stop.

        Returns:
            Span: A new extended span of the same type.

        Raises:
            AssertionError: If extension goes beyond study boundaries.
        """
        assert self.start >= t_before
        assert self.stop + t_after < self.study.tlen
        return type(self)(
            self.start - t_before,
            self.stop + t_after,
            self.study_id,
            self.study,
            self.metadata,
        )


class SpanSet:
    """A collection of Span objects, often representing a set of trials.

    Provides methods for batch operations across all spans in the set, such as
    extracting data, centering around events, or filtering based on metadata.

    Attributes:
        spans (list[Span]): The list of Span objects.
        df (pd.DataFrame, optional): Metadata for all spans as a DataFrame.
    """

    @property
    @abstractmethod
    def span_cls(self):
        """The class used to instantiate individual spans."""

    @property
    @abstractmethod
    def spanset_cls(self):
        """The class used to instantiate collections of spans."""

    @property
    @abstractmethod
    def spanarray_cls(self):
        """The class used to instantiate uniform arrays of spans."""

    def __init__(self, spans, **kwargs):
        """Initialize a new SpanSet collection.

        Args:
            spans (Iterable[Span]): A collection of Span objects to be
                managed as a group. All elements must be instances of
                the class defined in `span_cls`.
            **kwargs: Additional keyword arguments passed to parent classes.

        Raises:
            AssertionError: If any object in `spans` is not an instance of
                the expected `span_cls`.
        """
        super().__init__(**kwargs)
        assert all(isinstance(span, self.span_cls) for span in spans)

        self._spans = spans
        if self._spans and all(span.metadata is not None for span in spans):
            self._df = pd.DataFrame([span.metadata for span in spans])
        else:
            self._df = None

    @property
    def df(self):
        """Metadata DataFrame for all spans in the set."""
        return self._df

    @property
    def spans(self):
        """List of Span objects."""
        return self._spans

    def __iter__(self):
        """Iterate over spans in the set."""
        for span in self.spans:
            yield span

    def __len__(self):
        """Return the number of spans in the set."""
        return len(self.spans)

    def __getitem__(self, key):
        """Index into the set to retrieve spans or metadata columns."""
        # # metadata access
        if isinstance(key, str):
            return self.df[key]

        # single span
        if isinstance(key, (int, np.integer)):
            return self.spans[key]

        # sub-set
        if isinstance(key, slice):
            spans = self.spans[key]

        elif isinstance(key, Iterable):
            # boolean indexing
            if isinstance(key[0], (bool, np.bool_)):
                indices = np.where(key)[0]
                spans = [self.spans[idx] for idx in indices]
            # integer indexing
            elif isinstance(key[0], (int, np.integer)):
                spans = [self.spans[idx] for idx in key]
            else:
                raise ValueError("unsupported key type")

        else:
            raise ValueError("failed access")

        return self._make_spanset(spans)

    def _make_spanset(self, spans):
        """Create a new Spanset instance."""
        return self.spanset_cls(spans=spans)

    def _make_spanarray(self, spans):
        """Create a new SpanArray instance."""
        return self.spanarray_cls(spans=spans)

    def _wrap(self, func, *args, **kwargs):
        """Execute a method across all spans in the set."""
        return [getattr(span, func)(*args, **kwargs) for span in self]

    def around(self, t, t_before, t_after):
        """Align all spans in the set around a specific event or time."""
        span_arr = []
        for index, span in enumerate(self.spans):
            if isinstance(t, str):
                target_t = span[t]
            elif isinstance(t, Iterable):
                target_t = t[index]
            else:
                target_t = t
            span_arr.append(span.around(target_t, t_before, t_after))
        return self._make_spanarray(span_arr)

    def map(self, func, *args, **kwargs):
        """Apply a function to every span in the set."""
        out = []
        for span in self:
            out.append(func(span, *args, **kwargs))
        return out

    def refresh_metadata(self):
        """Synchronize span metadata with the central DataFrame."""
        assert self.df is not None, "no dataframe in `df` field"
        for span, row in zip(self.spans, self.df.index):
            span.metadata = self.df.loc[row]


class StudyMixin(ABC):
    """Mix-in defining the interface for a scientific study or session.

    A Study manages the top-level data access, including initializing metadata
    (DataFrames) and fetching raw or processed data files from remote storage.

    Attributes:
        study_id (str): Unique identifier for the study/session.
        subj_id (str): Identifier for the subject.
        url (str): Remote base URL for data fetching.
        download_dir (str): Local directory for data caching.
        initialized (bool): Whether the study has been initialized.
        quiet (bool): Whether to suppress progress bars and log messages.
    """

    # For linting purposes!
    span_cls: type

    def __init__(self, study_id, download_dir, quiet=False, **kwargs):
        """Initialize a new Study object.

        Args:
            study_id (str): Unique session identifier. The first character
                is extracted as the subject ID.
            download_dir (str): Local path where data files will be cached
                after being fetched from the remote endpoint.
            quiet (bool, optional): If True, suppresses tqdm progress bars
                and informational print statements during data fetching.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        self.study_id = study_id
        self.subj_id = self.study_id[0]
        self._download_dir = download_dir
        self.quiet = quiet
        self.initialized = False
        self._df = None
        self._spans = ()
        super().__init__(**kwargs)

    @property
    def df(self):
        """Metadata DataFrame. Accessing this triggers initialization."""
        if not self.initialized:
            self.initialize()
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def spans(self):
        """List of Span objects. Accessing this triggers initialization."""
        if not self.initialized:
            self.initialize()
        return self._spans

    @spans.setter
    def spans(self, value):
        self._spans = value

    def __getitem__(self, key):
        """Index into the study to retrieve spans."""
        if self.df is None:
            return self.by_time(key)
        return super().__getitem__(key)

    @cached_property
    @abstractmethod
    def tlen(self):
        """Return the total duration of the study in milliseconds."""

    @abstractmethod
    def by_time(self, key):
        """Retrieve a span by millisecond timestamps."""

    def __repr__(self):
        """Return string representation of the Study."""
        return f"study(study_id={self.study_id})"

    def initialize(self):
        """Initialize metadata and span objects."""
        if not self.initialized:
            self._init_meta()
            if self._df is not None:
                self._spans = [self._make_span(r) for _, r in self._df.iterrows()]
            else:
                self._spans = ()
            self.initialized = True
        return self

    @abstractmethod
    def _init_meta(self):
        """Initialize trial metadata DataFrame."""

    @abstractmethod
    def get_data(self, *names):
        """Fetch data from remote endpoint."""

    @abstractmethod
    def _make_span(self, row):
        """Create a span from a DataFrame row."""


class HeadH5Study(StudyMixin):
    """Base class for studies that use a 'head' HDF5 file for metadata/linking.

    This class provides common logic for managing HDF5 handles and fetching data
    using a predefined set of data paths.
    """

    data_paths = {}

    def __init__(self, study_id, download_dir="./data", quiet=False, **kwargs):
        """Initialize a head-based H5 study.

        Args:
            study_id (str): Unique session identifier.
            download_dir (str, optional): Local directory for caching data files.
                Defaults to "./data".
            quiet (bool, optional): Whether to suppress verbose output during
                fetching operations. Defaults to False.
            **kwargs: Additional keyword arguments passed to StudyMixin.
        """
        super().__init__(study_id, download_dir, quiet=quiet, **kwargs)
        self.head_handle = None
        self.has = set()
        self._url = None

    @cached_property
    @abstractmethod
    def fetcher(self) -> fetch.Fetcher:
        """Fetch manager."""

    @property
    def url(self) -> str:
        """Remote URL."""
        if self._url is not None:
            return self._url
        return self._default_url

    @url.setter
    def url(self, value):
        """Set remote URL and clear fetcher cache."""
        self._url = value
        self.__dict__.pop("fetcher", None)

    @property
    @abstractmethod
    def _default_url(self) -> str:
        """Default remote URL for the study."""

    @property
    def download_dir(self):
        """Local caching directory."""
        return self._download_dir

    @download_dir.setter
    def download_dir(self, value):
        """Set local caching directory and clear fetcher cache."""
        self._download_dir = value
        self.__dict__.pop("fetcher", None)

    @property
    def head(self) -> h5py.File:
        """Return the open HDF5 head handle, initializing if necessary."""
        if self.head_handle is None:
            # Explicit call replaced dynamic decorator
            head_path = self.get_data("head")[0]

            self.head_handle = h5py.File(
                head_path,
                "r",
                rdcc_nbytes=fetch.H5CACHE_RDCC_NBYTES,
                rdcc_nslots=fetch.H5CACHE_RDCC_NSLOTS,
            )
        return self.head_handle

    def get_data(self, *names):
        """Standard fetch-and-open implementation for Head-based studies."""
        return self.fetcher.get_data(self, *names)

    def _make_span(self, row):
        """Create a span from a DataFrame row.

        Args:
            row (pd.Series): Metadata row containing 'ms_start' and 'ms_end'.

        Returns:
            Span: A new span object representing the trial.
        """
        start = row["ms_start"]
        stop = row["ms_end"]
        # `span_cls` should generally be overriden upon merging
        # with a SpanSet implementation
        return self.span_cls(
            start=start, stop=stop, study_id=self.study_id, study=self, metadata=row
        )

    # ensure h5 is properly closed
    def __del__(self):
        """Ensure HDF5 handle is closed on deletion."""
        if hasattr(self, "head_handle") and self.head_handle is not None:
            self.head_handle.close()


class ArrayMixin:
    """Mix-in for SpanSets that return data as numpy arrays."""

    def __init__(self, spans, *args, **kwargs):
        """Initialize the ArrayMixin.

        Args:
            spans (Iterable[Span]): Collection of spans. All spans MUST
                have the same length (duration in milliseconds) to
                allow stacking into a numpy array.
            *args: Variable length argument list passed to parent.
            **kwargs: Arbitrary keyword arguments passed to parent.

        Raises:
            ValueError: If the provided spans do not have uniform lengths.
        """
        # check uniformity
        lengths = [len(s) for s in spans]
        if len(set(lengths)) != 1:
            raise ValueError("SpanArray requires spans of the same length")

        # init
        super().__init__(spans=spans, *args, **kwargs)

    def _wrap(self, func, *args, **kwargs):
        """Wrap wrapped results into a single numpy array."""
        return np.array(super()._wrap(func, *args, **kwargs))


class DataCatalog:
    """Registry for semantic data access methods."""

    def _return_data(self, func, **kwargs):
        """Helper to route data requests based on object type."""
        if isinstance(self, SpanSet):
            return self._wrap(func, **kwargs)
        if isinstance(self, Span):
            return self._data(func, **kwargs)
        return None
