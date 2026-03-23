import os
import shutil
import pandas as pd
import pytest

# pylint: disable=redefined-outer-name

from bil.api import get
from bil.api.abstracts import (
    sdr_url,
    SDR_URL,
    SDR_VERSIONED_URL,
    SpanSet,
    Span,
    StudyMixin,
)
from bil.api.utils.subject import Subject, UTAH
from bil.api.utils.fetch import FetcherHTTPS
from bil.api.utils import pickle as bil_pickle

# --- Configuration for Tests ---
TEST_DOWNLOAD_DIR = "tmp/test_download"


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Create and cleanup test download directory."""
    if os.path.exists(TEST_DOWNLOAD_DIR):
        shutil.rmtree(TEST_DOWNLOAD_DIR, ignore_errors=True)
    os.makedirs(TEST_DOWNLOAD_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DOWNLOAD_DIR):
        shutil.rmtree(TEST_DOWNLOAD_DIR, ignore_errors=True)


# --- Core Utilities & Abstracts Tests ---


def test_sdr_url_default():
    """Confirm default SDR URL is returned when no version is provided."""
    assert sdr_url() == SDR_URL


def test_sdr_url_versioned_int():
    """Confirm versioned SDR URL works with integer version."""
    version = 2
    expected = SDR_VERSIONED_URL.format(version=version)
    assert sdr_url(version) == expected


def test_sdr_url_versioned_str():
    """Confirm versioned SDR URL works with string version."""
    version = "3"
    expected = SDR_VERSIONED_URL.format(version=version)
    assert sdr_url(version) == expected


def test_study_invalid_id():
    """Confirm that an invalid study ID raises an appropriate error."""
    with pytest.raises(Exception):
        get("invalid_id", quiet=True).initialize()


# --- formatS Tests (using a known small public session) ---


@pytest.fixture
def study_s():
    """Fixture for a standard Rig S public study."""
    # U201130_01 is a known small session
    return get("U201130_01", download_dir=TEST_DOWNLOAD_DIR, quiet=True)


def test_study_s_initialization(study_s):
    """Confirm Study S initializes and populates metadata."""
    study_s.initialize()
    assert isinstance(study_s.df, pd.DataFrame)
    assert len(study_s.df) > 0
    assert hasattr(study_s, "spans")


def test_study_s_spans_count(study_s):
    """Confirm spans are correctly populated."""
    assert len(study_s.spans) > 0


def test_study_s_getitem(study_s):
    """Confirm integer indexing returns a Span."""
    span = study_s.spans[0]
    assert isinstance(span, Span)


def test_study_s_slice(study_s):
    """Confirm slicing returns a SpanSet."""
    subset = study_s[:2]
    assert isinstance(subset, SpanSet)
    assert len(subset) == 2


def test_study_s_boolean_indexing(study_s):
    """Confirm boolean indexing on spans."""
    study_s.initialize()
    if "number" in study_s.df.columns:
        mask = (study_s.df["number"] == 1).values
        subset = study_s[mask]
        assert len(subset) >= 0


def test_span_properties(study_s):
    """Confirm basic span properties."""
    span = study_s.spans[0]
    assert span.start < span.stop
    assert span.start == span["ms_start"]
    assert span.stop == span["ms_end"]


def test_span_around(study_s):
    """Confirm temporal alignment using around()."""
    span = study_s.spans[0]
    center = (span.stop - span.start) // 2
    new_span = span.around(center, 100, 100)
    assert new_span.stop - new_span.start == 201


def test_spanset_map(study_s):
    """Confirm SpanSet.map functionality."""
    durations = study_s[:3].map(lambda s: s.stop - s.start)
    assert len(durations) == 3
    assert all(d > 0 for d in durations)


def test_subject_mapping():
    """Test electrode array geometry mapping."""
    subj = Subject("U", arrays=[UTAH], regions=["m1"])
    assert subj.name == "U"
    assert hasattr(subj, "arrays")


def test_fetcher_https_url():
    """Confirm FetcherHTTPS constructs correct URLs."""
    f = FetcherHTTPS(SDR_URL, TEST_DOWNLOAD_DIR)
    assert f.base_url == SDR_URL


def test_pickle_utils():
    """Test custom pickling logic."""

    class MockStudy(StudyMixin):
        @property
        def tlen(self):
            return 100

        def _init_meta(self):
            pass

        def _make_span(self, row):
            return None

        def by_time(self, key):
            return None

        def get_data(self, *names):
            return []

        @property
        def fetcher(self):
            return None

        @property
        def url(self):
            return "http://test"

    data = MockStudy(study_id="U", download_dir=".")
    data = bil_pickle.pickle_proof(data)
    pickled = bil_pickle.pickle_down(data)
    unpickled = bil_pickle.pickle_up(pickled)
    assert unpickled.study_id == data.study_id


def test_abstract_study_mixin_init():
    """Test StudyMixin basic initialization without full Study."""

    class MockStudy(StudyMixin):
        @property
        def tlen(self):
            return 100

        def _init_meta(self):
            pass

        def _make_span(self, row):
            return None

        def by_time(self, key):
            return None

        def get_data(self, *names):
            return []

        @property
        def fetcher(self):
            return None

        @property
        def url(self):
            return "http://test"

        def initialize(self):
            self.initialized = True

    ms = MockStudy(study_id="test", download_dir=".")
    assert ms.study_id == "test"


def test_span_set_iteration(study_s):
    """Test iterating over SpanSet."""
    count = 0
    for _ in study_s.spans[:5]:
        count += 1
    assert count == 5


def test_span_set_empty():
    """Test empty SpanSet behavior."""
    ss = SpanSet(study=None, spans=[])
    assert len(ss) == 0
    with pytest.raises(IndexError):
        _ = ss[0]


def test_span_set_addition(study_s):
    """Test adding SpanSets together."""
    s1 = study_s.spans[:2]
    s2 = study_s.spans[2:4]
    combined = s1 + s2
    assert len(combined) == 4
    assert combined[0] == s1[0]
    assert combined[2] == s2[0]


def test_study_cache_invalidation():
    """Confirm that changing download_dir invalidates fetcher."""
    study = get("U201130_01", download_dir="tmp/dir1", quiet=True)
    f1 = study.fetcher
    study.download_dir = "tmp/dir2"
    assert study.fetcher is not f1
    assert str(study.fetcher.download_dir) == "tmp/dir2"


def test_study_quiet_propagation():
    """Confirm quiet flag propagates to fetcher."""
    study = get("U201130_01", quiet=True)
    assert study.fetcher.quiet is True
    study_loud = get("U201130_01", quiet=False)
    assert study_loud.fetcher.quiet is False


def test_span_tlen_logic(study_s):
    """Test span length logic."""
    span = study_s[-1]
    assert study_s.tlen >= span.stop


def test_span_metadata_access(study_s):
    """Test accessing metadata from a span."""
    span = study_s.spans[0]
    assert isinstance(span.metadata, pd.Series)


def test_study_getitem_errs(study_s):
    """Test string indexing on SpanSet if supported (should fail if not)."""
    # This checks if the error handling is correct
    with pytest.raises(ValueError):
        _ = study_s[3.0]
    with pytest.raises(KeyError):
        _ = study_s["invalid-key"]
