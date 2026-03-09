"""
API implementation for data format A.
"""

import sqlite3
import warnings
from functools import cached_property
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from . import abstracts
from .abstracts import needs_data
from .utils import fetch
from .utils import subject

# Channel mapping for Utah arrays, accounting for spatial orientation
U_M1_MAP = subject.UTAH_MAP.copy()[::-1, ::-1].T
H_M1_MAP = U_M1_MAP.copy()

# Substitutions for specific array configurations
U_PMD_MAP = U_M1_MAP.copy()
U_PMD_MAP[0, 0] = 95
U_PMD_MAP[2, 0] = np.nan
U_PMD_MAP[8, 0] = np.nan
U_PMD_MAP[9, 1] = np.nan
U_PMD_MAP[0, 9] = 89
U_PMD_MAP[9, 9] = 79

O_M1_MAP = U_M1_MAP.copy()
O_M1_MAP[4, 0] = np.nan
O_M1_MAP[0, 0] = 93

O_PMD_MAP = U_M1_MAP.copy()

# Registry of subjects and their associated arrays
SUBJECTS = {
    "U": subject.Subject(
        "U",
        [subject.UtahArray(U_M1_MAP), subject.UtahArray(U_PMD_MAP)],
        ["m1", "pmd"],
    ),
    "H": subject.Subject("H", [subject.UtahArray(H_M1_MAP)], ["m1"]),
    "O": subject.Subject(
        "O",
        [subject.UtahArray(O_M1_MAP), subject.UtahArray(O_PMD_MAP)],
        ["m1", "pmd"],
    ),
}


def get_subject(study_id):
    """Retrieve Subject object based on the study ID's subject prefix.

    Args:
        study_id (str): Unique identifier for the study.

    Returns:
        subject.Subject: The matched Subject object.
    """
    return SUBJECTS[study_id[0]]


class DataCatalog(abstracts.DataCatalog):
    """Catalog of high-level semantic data access methods for format A.

    These are high-level semantic methods for retrieving neural and
    behavioral signals from a span of time.
    """

    def lfp(self, region=None):
        """Retrieve Local Field Potential (LFP) data.

        [Units: Microvolts (1-250 Hz)]
        [Sampling Rate: 1000 Hz]

        Args:
            region (str, optional): Semantic brain region (e.g., 'm1').

        Returns:
            np.ndarray: LFP data of shape (channels, time).
        """
        return self._return_data("lfp", region=region)

    def raster(self, region=None):
        """Retrieve spike raster data (binned spikes).

        [Units: binary count per bin]
        [Sampling Rate: 1000 Hz]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Raster data of shape (channels, time).
        """
        return self._return_data("raster", region=region)

    def raw(self, region=None):
        """Retrieve raw wideband neural data.

        [Units: Microvolts]
        [Sampling Rate: 30,000 Hz]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Raw data of shape (channels, time).
        """
        return self._return_data("raw", region=region)

    def raster_30k(self, region=None):
        """Retrieve spike raster data at original sampling resolution.

        [Units: binary count per bin]
        [Sampling Rate: 30,000 Hz]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: High-resolution raster data.
        """
        return self._return_data("raster_30k", region=region)

    def raw_ch(self, ch):
        """Retrieve raw wideband data for a specific channel.

        [Units: Microvolts]
        [Sampling Rate: 30,000 Hz]

        Args:
            ch (int): 1-indexed channel number.

        Returns:
            np.ndarray: 1D array of raw voltage values.
        """
        return self._return_data("raw_ch", ch=ch)

    def sbp(self, region=None):
        """Retrieve Spiking Band Power (SBP).

        [Units: Microvolts (150-450 Hz)]
        [Sampling Rate: 1000 Hz]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: SBP data.
        """
        return self._return_data("sbp", region=region)

    def raster_rms(self, rms_threshold, region=None):
        """Retrieve spike raster filtered by an RMS noise threshold.

        [Units: binary count per bin]
        [Sampling Rate: 1000 Hz]

        Args:
            rms_threshold (float): Relative RMS threshold for excluding noisy units.
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Filtered raster data.
        """
        return self._return_data(
            "raster_rms", rms_threshold=rms_threshold, region=region
        )

    def valid(self, region=None):
        """Retrieve boolean mask indicating valid (non-dropped) data samples.

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Boolean array of shape (time,).
        """
        return self._return_data("valid", region=region)

    def pos_target(self):
        """Retrieve X and Y coordinates of the visual target.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Target position of shape (2, time).
        """
        return self._return_data("pos_target")

    def pos_hand(self):
        """Retrieve X, Y, and Z coordinates of the subject's hand/effector.

        [Units: mm (x, y, z)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Hand position of shape (3, time).
        """
        return self._return_data("pos_hand")

    def kinematics(self):
        """Retrieve cursor position or derived kinematic variables.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Kinematic data of shape (2, time).
        """
        return self._return_data("kinematics")

    def timestamp(self, region=None):
        """Retrieve precise hardware timestamps for each sample.

        [Units: Microseconds (us)]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: 1D array of timestamps.
        """
        return self._return_data("timestamp", region=region)

    def signal(self, signal_name):
        """Retrieve a generic signal by its HDF5 dataset name.

        Args:
            signal_name (str): Name of the signal in the HDF5 'signal' group.

        Returns:
            np.ndarray: Signal data.
        """
        return self._return_data("signal", signal_name=signal_name)

    def state_task(self):
        """Retrieve task state indices."""
        return self.signal("state_task")

    def tag_num(self):
        """Retrieve numeric tags associated with trials."""
        return self.signal("tag_num")

    def task_play(self):
        """Retrieve task play boolean indicator."""
        return self.signal("task_play")


class Span(DataCatalog, abstracts.Span):
    """Span implementation for format A.

    Provides specialized methods for retrieving and indexing into format A
    neural data and behavioral signals.
    """

    def _stream(self, region=None):
        """Identify relevant neural data streams for a region.

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Unique stream identifiers.
        """
        ninfo = yaml.safe_load(self.study.head.attrs["neural_info"])
        if region is None:
            idxs = np.arange(1, ninfo["num_neural_ch_total"] + 1)
        else:
            idx_slice = self.study.subject.region_chs[region]
            idxs = np.arange(idx_slice.start, idx_slice.stop) + 1
        stream_dict = ninfo["array_map"]
        streams = np.unique([stream_dict[idx] for idx in idxs])
        return streams

    def _valid(self, region=None):
        """Check if data within the span is valid across relevant streams.

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            bool: True if all data samples in the span are valid.
        """
        streams = self._stream(region)
        if "valid" in self.study.head["neural"]:
            for stream in streams:
                self.study.get_data(f"neural/valid{stream}.h5")
                valid_stream = self.study.head["neural"]["valid"][f"valid{stream}"]
                if not all(valid_stream[self.start : self.stop]):
                    return False
            return True

        out = []
        for stream in streams:
            self.study.get_data(f"neural/sync{stream}.h5")
            sync_stream = self.study.head["neural"]["sync"][f"sync{stream}"]
            out.append(sync_stream[self.start - 1 : self.stop])
        sync_diff = np.diff(np.stack(out), axis=-1)
        if ((sync_diff < 29) | (sync_diff > 31)).any():
            return False
        return True

    def _30k_ix(self, region=None):
        """Map 1kHz millisecond indices to 30kHz sample indices.

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            tuple: (start_index, stop_index) at 30kHz.
        """
        streams = self._stream(region)
        if streams.size > 1:
            raise ValueError("must specify a stream")
        stream = streams[0]
        self.study.get_data(f"neural/sync{stream}.h5")
        if self.start == 0:
            start = self.study.head["neural"]["sync"][f"sync{stream}"][0] - 30
        else:
            start = self.study.head["neural"]["sync"][f"sync{stream}"][self.start - 1]
        stop = self.study.head["neural"]["sync"][f"sync{stream}"][self.stop]
        return start, stop

    def _chs_from_region(self, region=None):
        """Retrieve list of channel numbers for a semantic region.

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: List of 1-indexed channel numbers.
        """
        num_ch = yaml.safe_load(self.study.head.attrs["neural_info"])[
            "num_neural_ch_total"
        ]
        chs = np.arange(1, num_ch + 1)
        chs = self._region(chs, region)
        return chs

    def _region(self, out, region=None):
        """Filter a data array by region-specific channel slices.

        Args:
            out (np.ndarray): Data array to filter.
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Sliced array.
        """
        if region is None:
            return out
        return out[self.study.subject.region_chs[region]]

    def _get(self, h5_ds, region=None):
        """Slice and return 1kHz data from an HDF5 dataset.

        Args:
            h5_ds (h5py.Dataset): The dataset to read from.
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Sliced data.
        """
        return self._region(h5_ds[..., self.start : self.stop], region)

    def _get_30k(self, h5_ds, region=None, ch=None):
        """Slice and return 30kHz data, handling multi-file stream assembly.

        Args:
            h5_ds (h5py.Dataset or str): The dataset or 'raw' indicator.
            region (str, optional): Semantic brain region.
            ch (int, optional): Specific channel index.

        Returns:
            np.ndarray: Sliced 30kHz data.
        """
        # Must fetch each individual file first
        if h5_ds.name == "/neural/raw/30k" and ch is None:
            chs = self._chs_from_region(region)
            self._get_raw_ch(*chs)

        if ch is not None:
            region = self.study.subject.ch_region(ch)
            start, stop = self._30k_ix(region=region)
            return h5_ds[ch - 1, start:stop]
        if region is not None:
            start, stop = self._30k_ix(region=region)
            ch_slice = self.study.subject.region_chs[region]
            return h5_ds[ch_slice, start:stop]

        if self._stream(region).size > 1:
            warnings.warn(
                "did not specify region but regions generated from separate data streams"
            )
        out = []
        for _region in self.study.subject.regions:
            start, stop = self._30k_ix(region=_region)
            ch_slice = self.study.subject.region_chs[_region]
            out.append(h5_ds[ch_slice, start:stop])
        truncate = min(x.shape[-1] for x in out)
        out = np.row_stack([x[..., :truncate] for x in out])
        return out

    def _neural(self, key="raster", fs="1k", region=None):
        """Retrieve neural data based on frequency and region.

        Args:
            key (str): Dataset key (raster, lfp, raw).
            fs (str): Sampling rate ('1k' or '30k').
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Neural signal.
        """
        h5_ds = self.study.head["neural"][key][fs]
        if fs == "30k":
            return self._get_30k(h5_ds, region=region)
        if fs == "1k":
            return self._get(h5_ds, region=region)
        return None

    def _signal(self, key):
        """Retrieve behavioral signal data.

        Args:
            key (str): Signal dataset name.

        Returns:
            np.ndarray: Signal data.
        """
        h5_ds = self.study.head["signal"][key]
        return self._get(h5_ds)

    def _data(self, func, **kwargs):
        """Internal router for data retrieval functions.

        Args:
            func (str): Requested data function name.
            **kwargs: Arguments for the specific function.

        Returns:
            Requested data or None.
        """
        # 1k neural
        if func in ("raster", "lfp", "sbp"):
            self.study.get_data(func)
            return self._neural(key=func, fs="1k", region=kwargs["region"])

        # 30k neural
        if func == "raster_30k":
            self.study.get_data(func)
            return self._neural(key="raster", fs="30k", region=kwargs["region"])
        if func == "raw":
            return self._neural(key=func, fs="30k", region=kwargs["region"])

        # special neural
        if func == "raw_ch":
            return self._raw_ch(ch=kwargs["ch"])
        if func == "raster_rms":
            return self._raster_rms(
                region=kwargs["region"], rms_threshold=kwargs["rms_threshold"]
            )

        # signals that require some custom assembly
        if func == "kinematics":
            return self._kinematics()
        if func == "pos_target":
            return self._pos_target()
        if func == "pos_hand":
            return self._pos_hand()

        # generic 1d signal
        if func == "signal":
            signal_name = kwargs["signal_name"]
            # Property access handles head fetch
            assert signal_name in self.study.head["signal"]
            self.study.get_data(f"signal/{signal_name}.h5")
            return self._signal(signal_name)

        # valid
        if func == "valid":
            return self._valid(region=kwargs["region"])

        # timestamp
        if func == "timestamp":
            return self._timestamp(region=kwargs["region"])

        return None

    def _timestamp(self, region=None):
        """Retrieve precise hardware timestamps for each sample.

        [Units: Microseconds (us)]

        Args:
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: 1D array of timestamps.
        """
        start, stop = self._30k_ix(region=region)
        stream = self._stream(region=region)
        assert len(stream) == 1
        stream = stream[0]
        self.study.get_data(f"neural/raw_timestamp{stream}.h5")
        h5_ds = self.study.head["neural"]["timestamp"][f"timestamp{stream}"]
        return h5_ds[start:stop]

    @needs_data("pos_hand")
    def _pos_hand(self):
        """Assemble 3D hand position data.

        [Units: mm (x, y, z)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Hand position of shape (3, time).
        """
        pos_hand_x = self._signal("pos_hand_x")
        pos_hand_y = self._signal("pos_hand_y")
        pos_hand_z = self._signal("pos_hand_z")

        return np.stack((pos_hand_x, pos_hand_y, pos_hand_z), axis=0)

    @needs_data("kinematics")
    def _kinematics(self):
        """Assemble 2D cursor kinematic data.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Kinematic data of shape (2, time).
        """
        pos_cursor_x = self._signal("pos_cursor_x")
        pos_cursor_y = self._signal("pos_cursor_y")
        return np.stack((pos_cursor_x, pos_cursor_y), axis=0)

    @needs_data("pos_target")
    def _pos_target(self):
        """Assemble 2D target position data.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            np.ndarray: Target position of shape (2, time).
        """
        pos_target_x = self._signal("pos_target_x")
        pos_target_y = self._signal("pos_target_y")
        return np.stack((pos_target_x, pos_target_y), axis=0)

    @needs_data("spike_db")
    def _spikes_at_rms(self, rms_threshold):
        """Query the SQLite database for spikes meeting a noise threshold.

        Args:
            rms_threshold (float): Relative RMS threshold for excluding noisy units.

        Returns:
            np.ndarray: Spike indices (channel, time).
        """
        db_path = Path(self.study.download_dir) / "neural" / "spike.db"
        db = sqlite3.connect(db_path).cursor()
        sql_string = f"""
        SELECT channel, lico_index FROM spike
        WHERE lico_index >= {self.start}
        AND lico_index < {self.stop}
        AND relative_rms < {rms_threshold}
        """
        spike_index = np.array(list(zip(*db.execute(sql_string).fetchall())))
        spike_index[0] -= 1
        spike_index[1] -= self.start
        return spike_index

    @needs_data("raster", "spike_db")
    def _raster_rms(self, rms_threshold, region=None):
        """Compute a binned raster for units below a specific noise level.

        Args:
            rms_threshold (float): Relative RMS threshold for excluding noisy units.
            region (str, optional): Semantic brain region.

        Returns:
            np.ndarray: Raster data.
        """
        spike_index = self._spikes_at_rms(rms_threshold)
        chs = yaml.safe_load(self.study.head.attrs["neural_info"])[
            "num_neural_ch_total"
        ]
        out = np.zeros((chs, self.stop - self.start))
        out[spike_index[0], spike_index[1]] = 1
        return self._region(out, region)

    def _get_raw_ch(self, *chs):
        """Ensure a single wideband channel file is fetched.

        Args:
            ch (int): 1-indexed channel number.
        """
        data_paths = []
        for ch in chs:
            num_ch = yaml.safe_load(self.study.head.attrs["neural_info"])[
                "num_neural_ch_total"
            ]
            zfill = len(str(num_ch))
            ch_str = f"ch{ch:0{zfill}}"
            data_path = f"neural/raw_{ch_str}.h5"
            data_paths.append(data_path)
        self.study.get_data(*data_paths)

    def _raw_ch(self, ch):
        """Retrieve raw wideband voltage for a single channel.

        [Units: Microvolts]
        [Sampling Rate: 30,000 Hz]

        Args:
            ch (int): 1-indexed channel number.

        Returns:
            np.ndarray: Raw signal.
        """
        self._get_raw_ch(ch)
        return self._get_30k(self.study.head["neural"]["raw"]["30k"], ch=ch)


class SpanSet(DataCatalog, abstracts.SpanSet):
    """Collection of Span objects for format A."""

    @property
    def span_cls(self):
        """The class used to instantiate individual spans."""
        return Span

    @property
    def spanarray_cls(self):
        """The class used to instantiate uniform arrays of spans."""
        return SpanArray

    @property
    def spanset_cls(self):
        """The class used to instantiate collections of spans."""
        return SpanSet


class SpanArray(abstracts.ArrayMixin, SpanSet):
    """Array representation of format A Spans."""


class StudyBase(abstracts.HeadH5Study):
    """Base Study class for format A datasets.

    Defines file paths and core metadata management for recording sessions.
    """

    data_paths = {
        "head": ["{run}.h5"],
        "df": ["df/trial.df.xz"],
        "raster_30k": ["neural/raster.h5"],
        "raster": ["neural/raster_1k.h5"],
        "lfp": ["neural/lfp_1k.h5"],
        "sbp": ["neural/sbp_1k.h5"],
        "spike_db": ["neural/spike.db"],
        "kinematics": ["signal/pos_cursor_x.h5", "signal/pos_cursor_y.h5"],
        "pos_hand": [
            "signal/pos_hand_x.h5",
            "signal/pos_hand_y.h5",
            "signal/pos_hand_z.h5",
        ],
        "pos_target": ["signal/pos_target_x.h5", "signal/pos_target_y.h5"],
    }

    def __init__(self, study_id, download_dir, quiet=False, **kwargs):
        """Initialize the StudyBase for format A data.

        Args:
            study_id (str): Unique session identifier (e.g. 'U201201_01').
            download_dir (str): Local path where data files are cached.
            quiet (bool, optional): If True, suppresses progress output during
                fetching. Defaults to False.
            **kwargs: Additional keyword arguments passed to HeadH5Study.
        """
        super().__init__(study_id, download_dir, quiet=quiet, **kwargs)
        self.subject = get_subject(self.study_id)
        self.has_raw_chs = set()

    @cached_property
    def tlen(self):
        """Return the total duration of the study in ticks.

        Returns:
            int: Duration in milliseconds.
        """
        return int(self.head.attrs["num_ticks"])

    @cached_property
    def rms_threshold(self):
        """Return the default RMS noise threshold for this session.

        Returns:
            float: RMS threshold.
        """
        return float(self.head["neural"]["raster"].attrs["rms_threshold"])

    @cached_property
    def rms(self):
        """Return the calculated RMS noise level for each channel.

        Returns:
            np.ndarray: RMS levels.
        """
        return np.array(self.head["neural"]["raster"].attrs["rms"], dtype=np.float64)

    def by_time(self, key):
        """Index by relative ms time.

        Args:
            key (int or slice): Millisecond index or range.

        Returns:
            Span: Resulting span.
        """
        return abstracts.by_time_ms(self, key, Span)


class Study(StudyBase, SpanSet):
    """Implementation of Study object for format A using HTTPS fetching.

    This is the primary public interface for accessing Format A data.
    """

    base_url = abstracts.REMOTE_HTTPS_URL

    def __init__(self, study_id, download_dir="data", quiet=False, **kwargs):
        """Initialize a format A Study with HTTPS data fetching.

        Args:
            study_id (str): Unique session identifier.
            download_dir (str, optional): Root directory for local data
                caching. The session-specific subdirectory will be
                created automatically. Defaults to "data".
            quiet (bool, optional): Whether to suppress progress bars and
                logs. Defaults to False.
            **kwargs: Additional keyword arguments passed to StudyBase.
        """
        super().__init__(
            study_id=study_id,
            download_dir=f"{download_dir}/{study_id}",
            quiet=quiet,
            spans=(),
            **kwargs,
        )

    @cached_property
    def fetcher(self):
        """Fetch manager for HTTPS data."""
        return fetch.FetcherHTTPS(self.url, self.download_dir, quiet=self.quiet)

    @property
    def _default_url(self):
        """Remote base URL for the session.

        Returns:
            str: Session URL.
        """
        return f"{self.base_url}/{self.study_id}"

    def _init_meta(self):
        """Fetch and load trial metadata DataFrame."""
        if self._df is not None:
            return
        assert self.fetcher.check_file_exists(
            "df/trial.df.xz"
        ), f"trial information not available; does the dataset {self.study_id} exist?"
        path = self.fetcher.get_file("df/trial.df.xz")
        df = pd.read_pickle(path)
        df = df.sort_values(by="number").reset_index(drop=True)
        self.df = df
