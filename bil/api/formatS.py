"""
API implementation for data format A.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import sqlite3
import warnings
from functools import cached_property
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import h5py

from . import abstracts
from .abstracts import needs_data
from .utils import subject

if TYPE_CHECKING:
    from .abstracts import StudyMixin

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
SUBJECTS: dict[str, subject.Subject] = {
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


def get_subject(study_id: str) -> subject.Subject:
    """Retrieve Subject object based on the study ID's subject prefix.

    Args:
        study_id: Unique identifier for the study.

    Returns:
        The matched Subject object.
    """
    return SUBJECTS[study_id[0]]


class DataCatalog(abstracts.DataCatalog):
    """Catalog of high-level semantic data access methods for format A.

    These are high-level semantic methods for retrieving neural and
    behavioral signals from a span of time.
    """

    def lfp(self, region: str | None = None) -> np.ndarray:
        """Retrieve Local Field Potential (LFP) data.

        Args:
            region: Semantic brain region.

        Returns:
            LFP data of shape (channels, time).
        """
        return self._return_data("lfp", region=region)

    def raster(self, region: str | None = None) -> np.ndarray:
        """Retrieve spike raster data (binned spikes).

        Args:
            region: Semantic brain region.

        Returns:
            Raster data of shape (channels, time).
        """
        return self._return_data("raster", region=region)

    def raw(self, region: str | None = None) -> np.ndarray:
        """Retrieve raw wideband neural data.

        Args:
            region: Semantic brain region.

        Returns:
            Raw data of shape (channels, time).
        """
        return self._return_data("raw", region=region)

    def raster_30k(self, region: str | None = None) -> np.ndarray:
        """Retrieve spike raster data at original sampling resolution.

        Args:
            region: Semantic brain region.

        Returns:
            High-resolution raster data.
        """
        return self._return_data("raster_30k", region=region)

    def raw_ch(self, ch: int) -> np.ndarray:
        """Retrieve raw wideband data for a specific ch.

        [Units: Microvolts]
        [Sampling Rate: 30,000 Hz]

        Args:
            ch: 1-indexed ch number.

        Returns:
            1D array of raw voltage values.
        """
        return self._return_data("raw_ch", ch=ch)

    def sbp(self, region: str | None = None) -> np.ndarray:
        """Retrieve Spiking Band Power (SBP).

        [Units: Microvolts (150-450 Hz)]
        [Sampling Rate: 1000 Hz]

        Args:
            region: Semantic brain region.

        Returns:
            SBP data.
        """
        return self._return_data("sbp", region=region)

    def raster_rms(self, rms_threshold: float, region: str | None = None) -> np.ndarray:
        """Retrieve spike raster filtered by an RMS noise threshold.

        [Units: binary count per bin]
        [Sampling Rate: 1000 Hz]

        Args:
            rms_threshold: Relative RMS threshold for excluding noisy units.
            region: Semantic brain region.

        Returns:
            Filtered raster data.
        """
        return self._return_data(
            "raster_rms", rms_threshold=rms_threshold, region=region
        )

    def valid(self, region: str | None = None) -> np.ndarray | bool:
        """Retrieve boolean mask indicating valid (non-dropped) data samples.

        Args:
            region: Semantic brain region.

        Returns:
            Boolean array of shape (time,) or single boolean.
        """
        return self._return_data("valid", region=region)

    def pos_target(self) -> np.ndarray:
        """Retrieve X and Y coordinates of the visual target.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Target position of shape (2, time).
        """
        return self._return_data("pos_target")

    def pos_hand(self) -> np.ndarray:
        """Retrieve X, Y, and Z coordinates of the subject's hand/effector.

        [Units: mm (x, y, z)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Hand position of shape (3, time).
        """
        return self._return_data("pos_hand")

    def kinematics(self) -> np.ndarray:
        """Retrieve cursor position or derived kinematic variables.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Kinematic data of shape (2, time).
        """
        return self._return_data("kinematics")

    def timestamp(self, region: str | None = None) -> np.ndarray:
        """Retrieve precise hardware timestamps.

        Args:
            region: Semantic brain region.

        Returns:
            1D array of timestamps.
        """
        return self._return_data("timestamp", region=region)

    def signal(self, signal_name: str) -> np.ndarray:
        """Retrieve a generic signal by its HDF5 dataset name.

        Args:
            signal_name: Name of the signal in the HDF5 'signal' group.

        Returns:
            Signal data.
        """
        return self._return_data("signal", signal_name=signal_name)

    def state_task(self) -> np.ndarray:
        """Retrieve task state indices.

        Returns:
            1D array of task state indices.
        """
        return self.signal("state_task")

    def tag_num(self) -> np.ndarray:
        """Retrieve numeric tags associated with trials.

        Returns:
            1D array of numeric tags.
        """
        return self.signal("tag_num")

    def task_play(self) -> np.ndarray:
        """Retrieve task play boolean indicator.

        Returns:
            1D array of task play boolean values.
        """
        return self.signal("task_play")


class Span(DataCatalog, abstracts.Span):
    """Span implementation for format A.

    Provides specialized methods for retrieving and indexing into format A
    neural data and behavioral signals.
    """

    study: StudyMixin

    def _stream(self, region: str | None = None) -> np.ndarray:
        """Identify relevant neural data streams.

        Args:
            region: Semantic brain region.

        Returns:
            Unique stream identifiers.
        """
        neural_info = yaml.safe_load(self.study.head.attrs["neural_info"])
        if region is None:
            indices = np.arange(1, neural_info["num_neural_ch_total"] + 1)
        else:
            idx_slice = self.study.subject.region_chs[region.lower()]
            indices = np.arange(idx_slice.start, idx_slice.stop) + 1
        stream_map = neural_info["array_map"]
        streams = np.unique([stream_map[idx] for idx in indices])
        return streams

    def _valid(self, region: str | None = None) -> bool:
        """Check if data is valid across relevant streams.

        Args:
            region: Semantic brain region.

        Returns:
            True if all data samples in the span are valid.
        """
        streams = self._stream(region)
        if "valid" in self.study.head["neural"]:
            for stream in streams:
                self.study.get_data(f"neural/valid{stream}.h5")
                valid_stream = self.study.head["neural"]["valid"][f"valid{stream}"]
                if not all(valid_stream[self.start : self.stop]):
                    return False
            return True

        output = []
        for stream in streams:
            self.study.get_data(f"neural/sync{stream}.h5")
            sync_stream = self.study.head["neural"]["sync"][f"sync{stream}"]
            output.append(sync_stream[self.start - 1 : self.stop])
        sync_differences = np.diff(np.stack(output), axis=-1)
        if ((sync_differences < 29) | (sync_differences > 31)).any():
            return False
        return True

    def _ix_30k(self, region: str | None = None) -> tuple[int, int]:
        """Map 1kHz millisecond indices to 30kHz sample indices.

        Args:
            region: Semantic brain region.

        Returns:
            (start_index, stop_index) at 30kHz.
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

    def _chs_from_region(self, region: str | None = None) -> np.ndarray:
        """Retrieve list of channel numbers.

        Args:
            region: Semantic brain region.

        Returns: list of 1-indexed channel numbers.
        """
        num_channels = yaml.safe_load(self.study.head.attrs["neural_info"])[
            "num_neural_ch_total"
        ]
        channels = np.arange(1, num_channels + 1)
        channels = self._region(channels, region)
        return channels

    def _region(self, out: np.ndarray, region: str | None = None) -> np.ndarray:
        """Filter data array by region-specific channel slices.

        Args:
            out: Data array.
            region: Semantic brain region.

        Returns:
            Sliced array.
        """
        if region is None:
            return out
        return out[self.study.subject.region_chs[region.lower()]]

    def _get(self, h5_dataset: h5py.Dataset, region: str | None = None) -> np.ndarray:
        """Slice and return 1kHz data from an HDF5 dataset.

        Args:
            h5_dataset: Dataset handle.
            region: Semantic brain region.

        Returns:
            Sliced data.
        """
        return self._region(h5_dataset[..., self.start : self.stop], region)

    def _get_30k(
        self,
        h5_dataset: h5py.Dataset,
        region: str | None = None,
        channel: int | None = None,
    ) -> np.ndarray:
        """Slice and return 30kHz data.

        Args:
            h5_dataset: Dataset.
            region: Semantic brain region.
            channel: Specific channel index.

        Returns:
            Sliced 30kHz data.
        """
        # Must fetch each individual file first
        if (
            hasattr(h5_dataset, "name")
            and h5_dataset.name == "/neural/raw/30k"
            and channel is None
        ):
            channels = self._chs_from_region(region)
            self._get_raw_ch(*channels)

        if channel is not None:
            region = self.study.subject.ch_region(channel)
            start, stop = self._ix_30k(region=region)
            return h5_dataset[channel - 1, start:stop]
        if region is not None:
            start, stop = self._ix_30k(region=region)
            channel_slice = self.study.subject.region_chs[region]
            return h5_dataset[channel_slice, start:stop]

        if self._stream(region).size > 1:
            warnings.warn(
                "did not specify region but regions generated from separate data streams"
            )
        data = []
        for region_name in self.study.subject.regions:
            start, stop = self._ix_30k(region=region_name)
            channel_slice = self.study.subject.region_chs[region_name]
            data.append(h5_dataset[channel_slice, start:stop])
        truncate = min(x.shape[-1] for x in data)
        output = np.row_stack([x[..., :truncate] for x in data])
        return output

    def _neural(
        self,
        key: str = "raster",
        sampling_rate: str = "1k",
        region: str | None = None,
    ) -> np.ndarray | None:
        """Retrieve neural data based on frequency and region.

        Args:
            key: Dataset key.
            sampling_rate: Sampling rate ('1k' or '30k').
            region: Semantic brain region.

        Returns:
            Neural signal.
        """
        h5_dataset = self.study.head["neural"][key][sampling_rate]
        if sampling_rate == "30k":
            return self._get_30k(h5_dataset, region=region)
        if sampling_rate == "1k":
            return self._get(h5_dataset, region=region)
        return None

    def _signal(self, key: str) -> np.ndarray:
        """Retrieve behavioral signal data.

        Args:
            key: Signal dataset name.

        Returns:
            Signal data.
        """
        h5_dataset = self.study.head["signal"][key]
        return self._get(h5_dataset)

    def _data(self, func: str, **kwargs: Any) -> Any:
        """Internal router for data retrieval functions.

        Args:
            func: Requested data function name.
            **kwargs: Arguments.

        Returns:
            Requested data.
        """
        # 1k neural
        if func in ("raster", "lfp", "sbp"):
            self.study.get_data(func)
            return self._neural(
                key=func, sampling_rate="1k", region=kwargs.get("region")
            )

        # 30k neural
        if func == "raster_30k":
            self.study.get_data(func)
            return self._neural(
                key="raster", sampling_rate="30k", region=kwargs.get("region")
            )
        if func == "raw":
            return self._neural(
                key=func, sampling_rate="30k", region=kwargs.get("region")
            )

        # special neural
        if func == "raw_ch":
            return self._raw_ch(channel=kwargs["ch"])
        if func == "raster_rms":
            return self._raster_rms(
                region=kwargs.get("region"), rms_threshold=kwargs["rms_threshold"]
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
            return self._valid(region=kwargs.get("region"))

        # timestamp
        if func == "timestamp":
            return self._timestamp(region=kwargs.get("region"))

        return None

    def _timestamp(self, region: str | None = None) -> np.ndarray:
        """Retrieve hardware timestamps.

        Args:
            region: Semantic brain region.

        Returns:
            1D array of timestamps.
        """
        start, stop = self._ix_30k(region=region)
        streams = self._stream(region=region)
        assert len(streams) == 1
        stream = streams[0]
        self.study.get_data(f"neural/raw_timestamp{stream}.h5")
        h5_dataset = self.study.head["neural"]["timestamp"][f"timestamp{stream}"]
        return h5_dataset[start:stop]

    @needs_data("pos_hand")
    def _pos_hand(self) -> np.ndarray:
        """Assemble 3D hand position data.

        [Units: mm (x, y, z)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Hand position of shape (3, time).
        """
        pos_hand_x = self._signal("pos_hand_x")
        pos_hand_y = self._signal("pos_hand_y")
        pos_hand_z = self._signal("pos_hand_z")

        return np.stack((pos_hand_x, pos_hand_y, pos_hand_z), axis=0)

    @needs_data("kinematics")
    def _kinematics(self) -> np.ndarray:
        """Assemble 2D cursor kinematic data.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Kinematic data of shape (2, time).
        """
        pos_cursor_x = self._signal("pos_cursor_x")
        pos_cursor_y = self._signal("pos_cursor_y")
        return np.stack((pos_cursor_x, pos_cursor_y), axis=0)

    @needs_data("pos_target")
    def _pos_target(self) -> np.ndarray:
        """Assemble 2D target position data.

        [Units: mm (x, y)]
        [Sampling Rate: 1000 Hz]

        Returns:
            Target position of shape (2, time).
        """
        pos_target_x = self._signal("pos_target_x")
        pos_target_y = self._signal("pos_target_y")
        return np.stack((pos_target_x, pos_target_y), axis=0)

    @needs_data("spike_db")
    def _spikes_at_rms(self, rms_threshold: float) -> np.ndarray:
        """Query the SQLite database for spikes meeting a noise threshold.

        Args:
            rms_threshold: Relative RMS threshold for excluding noisy units.

        Returns:
            Spike indices (channel, time).
        """
        database_path = Path(self.study.download_dir) / "neural" / "spike.db"
        database = sqlite3.connect(database_path).cursor()
        sql_query = f"""
        SELECT channel, lico_index FROM spike
        WHERE lico_index >= {self.start}
        AND lico_index < {self.stop}
        AND relative_rms < {rms_threshold}
        """
        spike_indices = np.array(list(zip(*database.execute(sql_query).fetchall())))
        spike_indices[0] -= 1
        spike_indices[1] -= self.start
        return spike_indices

    @needs_data("raster", "spike_db")
    def _raster_rms(
        self, rms_threshold: float, region: str | None = None
    ) -> np.ndarray:
        """Compute a binned raster for units below a specific noise level.

        Args:
            rms_threshold: Relative RMS threshold for excluding noisy units.
            region: Semantic brain region.

        Returns:
            Raster data.
        """
        spike_indices = self._spikes_at_rms(rms_threshold)
        num_channels = yaml.safe_load(self.study.head.attrs["neural_info"])[
            "num_neural_ch_total"
        ]
        output = np.zeros((num_channels, self.stop - self.start))
        output[spike_indices[0], spike_indices[1]] = 1
        return self._region(output, region)

    def _get_raw_ch(self, *channels: int) -> None:
        """Ensure a single wideband channel file is fetched.

        Args:
            *channels: 1-indexed channel numbers.
        """
        data_paths = []
        for channel in channels:
            num_channels = yaml.safe_load(self.study.head.attrs["neural_info"])[
                "num_neural_ch_total"
            ]
            zero_fill = len(str(num_channels))
            channel_string = f"ch{channel:0{zero_fill}}"
            data_path = f"neural/raw_{channel_string}.h5"
            data_paths.append(data_path)
        self.study.get_data(*data_paths)

    def _raw_ch(self, channel: int) -> np.ndarray:
        """Retrieve raw wideband voltage for a single channel.

        [Units: Microvolts]
        [Sampling Rate: 30,000 Hz]

        Args:
            channel: 1-indexed channel number.

        Returns:
            Raw signal.
        """
        self._get_raw_ch(channel)
        return self._get_30k(self.study.head["neural"]["raw"]["30k"], channel=channel)


class SpanSet(DataCatalog, abstracts.SpanSet):
    """Collection of Span objects for format A."""

    @property
    def span_cls(self) -> type[Span]:
        """Span class."""
        return Span

    @property
    def spanarray_cls(self) -> type["SpanArray"]:
        """SpanArray class."""
        return SpanArray

    @property
    def spanset_cls(self) -> type["SpanSet"]:
        """SpanSet class."""
        return SpanSet


class SpanArray(abstracts.ArrayMixin, SpanSet):
    """Array representation of format A Spans."""


class StudyBase(abstracts.HeadH5Study):
    """Base Study class for format A datasets."""

    data_paths: dict[str, list[str]] = {
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

    def __init__(
        self, study_id: str, download_dir: str, quiet: bool = False, **kwargs: Any
    ) -> None:
        """Initialize StudyBase.

        Args:
            study_id: Session identifier.
            download_dir: Cache path.
            quiet: Verbosity flag.
            **kwargs: Additional options.
        """
        super().__init__(study_id, download_dir, quiet=quiet, **kwargs)
        self.subject = get_subject(self.study_id)
        self.has_raw_chs: set[int] = set()

    @cached_property
    def tlen(self) -> int:
        """Return the total duration of the study in ticks.

        Returns:
            Duration in milliseconds.
        """
        return int(self.head.attrs["num_ticks"])

    @cached_property
    def rms_threshold(self) -> float:
        """Return the default RMS noise threshold for this session.

        Returns:
            RMS threshold.
        """
        return float(self.head["neural"]["raster"].attrs["rms_threshold"])

    @cached_property
    def rms(self) -> np.ndarray:
        """Return the calculated RMS noise level for each channel.

        Returns:
            RMS levels.
        """
        return np.array(self.head["neural"]["raster"].attrs["rms"], dtype=np.float64)

    def by_time(self, key: int | slice) -> Span:
        """Index by relative ms time.

        Args:
            key: Time index or slice in milliseconds.

        Returns:
            A Span object.
        """
        return abstracts.by_time_ms(self, key, Span)


class Study(abstracts.PublicMixin, StudyBase, SpanSet):
    """Study object for format A using HTTPS fetching."""

    def _init_meta(self) -> None:
        """Fetch and load trial metadata."""
        if self._df is not None:
            return
        assert self.fetcher.check_file_exists(
            "df/trial.df.xz"
        ), f"trial information not available; does the dataset {self.study_id} exist?"
        path = self.fetcher.get_file("df/trial.df.xz")
        dataframe = pd.read_pickle(path)
        dataframe = dataframe.sort_values(by="number").reset_index(drop=True)
        self.df = dataframe
