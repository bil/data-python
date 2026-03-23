"""
Abstract base classes and core utilities for the BIL data API.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import h5py

from .utils import fetch

SDR_VERSIONED_URL: str = (
    "https://stacks.stanford.edu/v{version}/file/zz618yg1930/version/{version}/data"
)
SDR_URL: str = "https://stacks.stanford.edu/file/zz618yg1930/data"


def sdr_url(version: int | str | None = None) -> str:
    """Format SDR URL with an optional version.

    Args:
        version: Deposition version.

    Returns:
        Formatted URL.
    """
    if version is None:
        return SDR_URL

    return SDR_VERSIONED_URL.format(version=int(version))


def needs_data(*names: str) -> Callable:
    """Decorator ensuring specific data files are downloaded before execution.

    Args:
        *names: Names of data files required.

    Returns:
        Decorator function.
    """

    def inner(func: Callable) -> Callable:
        """Inner decorator factory.

        Args:
            func: Function to decorate.

        Returns:
            Wrapped function.
        """

        @functools.wraps(func)
        def decorator(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapper fetching data before execution."""
            if hasattr(self, "study"):
                self.study.get_data(*names)
            elif hasattr(self, "get_data"):
                self.get_data(*names)
            else:
                raise RuntimeError("trying to decorate invalid object")

            return func(self, *args, **kwargs)

        return decorator

    return inner


def by_time_ms(study: Any, key: int | slice, span_cls: type["Span"]) -> "Span":
    """Standard implementation of by_time indexing in milliseconds.

    Args:
        study: Parent study object.
        key: Time index or range.
        span_cls: Span class to instantiate.

    Returns:
        New span object.

    Raises:
        IndexError: If time is out of bounds.
        ValueError: If key type is unsupported.
    """
    if isinstance(key, (int, np.integer)):
        if not 0 <= key <= study.tlen:
            raise IndexError(f"key {key} out of bounds [0, {study.tlen}]")
        start = int(key)
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

    Attributes:
        start: Start time (inclusive).
        stop: Stop time (exclusive).
        study_id: Study identifier.
        study: Parent Study object.
        metadata: Associated metadata.
    """

    def __init__(
        self,
        start: int,
        stop: int,
        study_id: str,
        study: Any,
        metadata: dict | pd.Series | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Span.

        Args:
            start: Start time in ms.
            stop: Stop time in ms (exclusive).
            study_id: Session identifier.
            study: Reference to parent Study object.
            metadata: Trial metadata.
            **kwargs: Additional arguments.
        """
        self.start = start
        self.stop = stop
        self.study_id = study_id
        self.study = study
        self.metadata = metadata
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"span(study_id={self.study_id}, start={self.start}ms, stop={self.stop}ms)"
        )

    def __getitem__(self, key: str | int | slice) -> Any:
        """Index into span for metadata or sub-span.

        Args:
            key: Metadata key or temporal index.

        Returns:
            Metadata value or sub-span.

        Raises:
            ValueError: Unsupported key type.
            KeyError: Missing metadata.
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

    def __len__(self) -> int:
        """Return duration in ms."""
        return self.stop - self.start

    @abstractmethod
    def _data(self, func: Callable, **kwargs: Any) -> Any:
        """Internal data retrieval."""

    def around(self, time_ms: int, t_before: int, t_after: int) -> "Span":
        """Create new span centered around a time point.

        Args:
            time_ms: Center time relative to span start.
            t_before: Duration before center.
            t_after: Duration after center.

        Returns:
            New aligned span.
        """
        start = time_ms - t_before
        end = time_ms + t_after + 1

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

    def extend(self, t_before: int = 0, t_after: int = 0) -> "Span":
        """Return extended span.

        Args:
            t_before: Amount to extend backward.
            t_after: Amount to extend forward.

        Returns:
            Extended span.
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
    """Collection of Span objects.

    Attributes:
        spans: list of Spans.
        df: Metadata for all spans.
    """

    @property
    @abstractmethod
    def span_cls(self) -> type[Span]:
        """Span instantiation class."""

    @property
    @abstractmethod
    def spanset_cls(self) -> type["SpanSet"]:
        """SpanSet instantiation class."""

    @property
    @abstractmethod
    def spanarray_cls(self) -> type["SpanSet"]:
        """SpanArray instantiation class."""

    def __init__(
        self, spans: Iterable[Span], **kwargs: Any
    ) -> None:  # pylint: disable=unused-argument
        """Initialize SpanSet.

        Args:
            spans: Collection of spans.
            **kwargs: Additional arguments.
        """
        super().__init__()
        assert all(isinstance(span, self.span_cls) for span in spans)

        self._spans = list(spans)
        if self._spans and all(span.metadata is not None for span in spans):
            self._df = pd.DataFrame([span.metadata for span in spans])
        else:
            self._df = None

    @property
    def df(self) -> pd.DataFrame | None:
        """Metadata DataFrame for all spans in the set."""
        return self._df

    @property
    def spans(self) -> list[Span]:
        """List of Span objects."""
        return self._spans

    def __iter__(self) -> Iterable[Span]:
        """Iterate over spans."""
        for span in self.spans:
            yield span

    def __len__(self) -> int:
        """Return number of spans."""
        return len(self.spans)

    def __getitem__(self, key: str | int | slice | Iterable) -> Any:
        """Index into set for spans or metadata.

        Args:
            key: Index or key.

        Returns:
            Span, SpanSet, or metadata column.
        """
        # # metadata access
        if isinstance(key, str):
            if self.df is None:
                raise KeyError("Dataframe is not initialized")
            return self.df[key]

        # single span
        if isinstance(key, (int, np.integer)):
            return self.spans[int(key)]

        # sub-set
        if isinstance(key, slice):
            spans = self.spans[key]

        elif isinstance(key, Iterable):
            key_list = list(key)
            # boolean indexing
            if isinstance(key_list[0], (bool, np.bool_)):
                indices = np.where(key_list)[0]
                spans = [self.spans[int(idx)] for idx in indices]
            # integer indexing
            elif isinstance(key_list[0], (int, np.integer)):
                spans = [self.spans[int(idx)] for idx in key_list]
            else:
                raise ValueError("unsupported key type")

        else:
            raise ValueError("failed access")

        return self._make_spanset(spans)

    def _make_spanset(self, spans: list[Span]) -> "SpanSet":
        """Create new Spanset.

        Args:
            spans: list of spans.

        Returns:
            New SpanSet.
        """
        return self.spanset_cls(spans=spans)

    def _make_spanarray(self, spans: list[Span]) -> "SpanSet":
        """Create new SpanArray.

        Args:
            spans: list of spans.

        Returns:
            New SpanArray.
        """
        return self.spanarray_cls(spans=spans)

    def _wrap(self, func: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Execute method across all spans.

        Args:
            func: Method name to call on each span.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns: list of results.
        """
        return [getattr(span, func)(*args, **kwargs) for span in self]

    def around(
        self, time_ms: str | int | Iterable, t_before: int, t_after: int
    ) -> "SpanSet":
        """Align all spans around an event.

        Args:
            time_ms: Target time.
            t_before: Duration before.
            t_after: Duration after.

        Returns:
            New SpanArray.
        """
        span_arr = []
        for index, span in enumerate(self.spans):
            if isinstance(time_ms, str):
                target_t = span[time_ms]
            elif isinstance(time_ms, Iterable):
                target_t = list(time_ms)[index]
            else:
                target_t = time_ms
            span_arr.append(span.around(int(target_t), t_before, t_after))
        return self._make_spanarray(span_arr)

    def map(self, func: Callable, *args: Any, **kwargs: Any) -> list[Any]:
        """Apply function to every span.

        Args:
            func: Function to apply.
            *args: Additional arguments for func.
            **kwargs: Additional keyword arguments for func.

        Returns: list of results.
        """
        out = []
        for span in self:
            out.append(func(span, *args, **kwargs))
        return out

    def refresh_metadata(self) -> None:
        """Synchronize span metadata with central DataFrame."""
        assert self.df is not None, "no dataframe in `df` field"
        for span, row in zip(self.spans, self.df.index):
            span.metadata = self.df.loc[row]


class StudyMixin(ABC):
    """Mix-in for scientific study interface.

    Attributes:
        study_id: Unique session identifier.
        subj_id: Subject identifier.
        url: Remote base URL.
        download_dir: Local caching directory.
        initialized: Initialization status.
        quiet: Verbosity flag.
    """

    # For linting purposes!
    span_cls: type[Span]

    def __init__(
        self, study_id: str, download_dir: str, quiet: bool = False, **kwargs: Any
    ) -> None:
        """Initialize Study.

        Args:
            study_id: Session identifier.
            download_dir: Cache directory.
            quiet: Suppress output.
            **kwargs: Additional arguments.
        """
        self.study_id = study_id
        self.subj_id = self.study_id[0]
        self._download_dir = download_dir
        self.quiet = quiet
        self.initialized = False
        self._df: pd.DataFrame | None = None
        self._spans: list[Span] | tuple = ()
        super().__init__(**kwargs)

    @property
    def df(self) -> pd.DataFrame | None:
        """Metadata DataFrame. Accessing triggers initialization."""
        if not self.initialized:
            self.initialize()
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame | None) -> None:
        self._df = value

    @property
    def spans(self) -> list[Span] | tuple:
        """List of Span objects. Accessing triggers initialization."""
        if not self.initialized:
            self.initialize()
        return self._spans

    @spans.setter
    def spans(self, value: list[Span] | tuple) -> None:
        self._spans = value

    def __getitem__(self, key: Any) -> Any:
        """Index into study to retrieve spans.

        Args:
            key: Index or key.

        Returns:
            Result of indexing.
        """
        if self.df is None:
            return self.by_time(key)
        return super().__getitem__(key)

    @cached_property
    @abstractmethod
    def tlen(self) -> int:
        """Total duration in ms."""

    @abstractmethod
    def by_time(self, key: int | slice) -> Span:
        """Retrieve span by millisecond timestamps.

        Args:
            key: Time index or range.

        Returns:
            New span object.
        """

    def __repr__(self) -> str:
        """Return string representation."""
        return f"study(study_id={self.study_id})"

    def initialize(self) -> "StudyMixin":
        """Initialize metadata and spans.

        Returns:
            Initialized study.
        """
        if not self.initialized:
            self._init_meta()
            if self._df is not None:
                self._spans = [self._make_span(row) for _, row in self._df.iterrows()]
            else:
                self._spans = ()
            self.initialized = True
        return self

    @abstractmethod
    def _init_meta(self) -> None:
        """Initialize trial metadata."""

    @abstractmethod
    def get_data(self, *names: str) -> list[str]:
        """Fetch data from remote endpoint.

        Args:
            *names: Data file names.

        Returns:
            Paths to fetched files.
        """

    @abstractmethod
    def _make_span(self, row: pd.Series) -> Span:
        """Create span from DataFrame row.

        Args:
            row: Metadata row.

        Returns:
            New span.
        """


class HeadH5Study(StudyMixin):
    """Base class for studies with remote HDF5 head files."""

    data_paths: dict[str, list[str]] = {}

    def __init__(
        self,
        study_id: str,
        download_dir: str = "./data",
        quiet: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize head-based H5 study.

        Args:
            study_id: Session identifier.
            download_dir: Cache directory.
            quiet: Suppress output.
            **kwargs: Additional arguments.
        """
        super().__init__(study_id, download_dir, quiet=quiet, **kwargs)
        self.head_handle: h5py.File | None = None
        self.has: set = set()
        self._url: str | None = None

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
    def url(self, value: str) -> None:
        """Set remote URL and clear fetcher."""
        self._url = value
        self.__dict__.pop("fetcher", None)

    @property
    @abstractmethod
    def _default_url(self) -> str:
        """Default remote URL."""

    @property
    def download_dir(self) -> str:
        """Local caching directory."""
        return self._download_dir

    @download_dir.setter
    def download_dir(self, value: str) -> None:
        """Set local caching directory and clear fetcher."""
        self._download_dir = value
        self.__dict__.pop("fetcher", None)

    @property
    def head(self) -> h5py.File:
        """Return open HDF5 head handle."""
        if self.head_handle is None:
            head_path = self.get_data("head")[0]

            self.head_handle = h5py.File(
                head_path,
                "r",
                rdcc_nbytes=fetch.H5CACHE_RDCC_NBYTES,
                rdcc_nslots=fetch.H5CACHE_RDCC_NSLOTS,
            )
        return self.head_handle

    def get_data(self, *names: str) -> list[str]:
        """Standard fetch-and-open implementation.

        Args:
            *names: Data file names.

        Returns:
            Paths to fetched files.
        """
        return self.fetcher.get_data(self, *names)

    def _make_span(self, row: pd.Series) -> Span:
        """Create span from DataFrame row.

        Args:
            row: Metadata row.

        Returns:
            New span.
        """
        start = int(row["ms_start"])
        stop = int(row["ms_end"])
        return self.span_cls(
            start=start, stop=stop, study_id=self.study_id, study=self, metadata=row
        )

    # ensure h5 is properly closed
    def __del__(self) -> None:
        """Ensure HDF5 handle is closed."""
        if hasattr(self, "head_handle") and self.head_handle is not None:
            self.head_handle.close()


class PublicMixin:
    """Mix-in for public HTTPS data access."""

    study_id: str

    def __init__(
        self,
        study_id: str,
        download_dir: str = "data",
        quiet: bool = False,
        deposition_version: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize format A Study with HTTPS fetching.

        Args:
            study_id: Session identifier.
            download_dir: Cache directory.
            quiet: Suppress output.
            deposition_version: Deposition version.
            **kwargs: Additional arguments.
        """
        self.deposition_version = deposition_version
        super().__init__(
            study_id=study_id,
            download_dir=f"{download_dir}/{study_id}",
            quiet=quiet,
            spans=(),
            **kwargs,
        )

    @cached_property
    def fetcher(self) -> fetch.FetcherHTTPS:
        """HTTPS fetch manager."""
        return fetch.FetcherHTTPS(self.url, self.download_dir, quiet=self.quiet)

    @property
    def _default_url(self) -> str:
        """Session URL."""
        return f"{sdr_url(self.deposition_version)}/{self.study_id}"


class ArrayMixin:
    """Mix-in for SpanSets returning numpy arrays."""

    def __init__(self, spans: Iterable[Span], *args: Any, **kwargs: Any) -> None:
        """Initialize ArrayMixin.

        Args:
            spans: Collection of spans.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        spans_list = list(spans)
        # check uniformity
        lengths = [len(span) for span in spans_list]
        if len(set(lengths)) != 1:
            raise ValueError("SpanArray requires spans of the same length")

        # init
        super().__init__(spans=spans_list, *args, **kwargs)

    def _wrap(self, func: str, *args: Any, **kwargs: Any) -> np.ndarray:
        """Wrap results into numpy array.

        Args:
            func: Method name to call on each span.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Numpy array of results.
        """
        return np.array(super()._wrap(func, *args, **kwargs))


class DataCatalog:
    """Registry for semantic data access methods."""

    def _return_data(self, func: str, **kwargs: Any) -> Any:
        """Route data requests.

        Args:
            func: Method name.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of routing.
        """
        if isinstance(self, SpanSet):
            return self._wrap(func, **kwargs)
        if isinstance(self, Span):
            return self._data(func, **kwargs)
        return None
