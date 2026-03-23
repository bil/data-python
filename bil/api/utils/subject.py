"""
Representations of experimental subjects and microelectrode arrays (MEAs).

This module defines classes for managing subject-specific metadata, brain regions,
and the spatial configuration of arrays.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np

UTAH_MAP = np.concatenate(
    (
        [np.nan],
        np.arange(1, 8 + 1),
        [np.nan],
        np.arange(9, 88 + 1),
        [np.nan],
        np.arange(89, 96 + 1),
        [np.nan],
    )
).reshape((10, 10))


class MEA:
    """Base class for Multielectrode Arrays.

    Handles coordinate mapping and pairwise distance calculations for
    electrode sites.

    Attributes:
        ch_coords: (ch, dim) set of spatial coordinates for each channel.
        dist: Distance metric ('euclidean' or 'manhattan').
        pw_distance: Precomputed pairwise distance matrix.
        num_ch: Number of channels in the array.
    """

    def __init__(self, locs: np.ndarray | Iterable, dist: str = "euclidean") -> None:
        """Initialize a Multielectrode Array (MEA).

        Args:
            locs: Spatial coordinates for each electrode site.
                Typically an (N, 2) or (N, 3) array-like object where N
                is the number of channels.
            dist: Distance metric used for pairwise
                calculations. Supported: 'euclidean', 'manhattan'.
                Defaults to "euclidean".
        """
        self.ch_coords = np.array(locs)
        self.dist = dist
        self.pw_distance = self._pw_distance()
        self.num_ch = len(self.ch_coords)

    def _pw_distance(self) -> np.ndarray:
        """Compute pairwise distances between all channels.

        Returns:
            A square matrix of pairwise distances.
        """
        distances = self.ch_coords[:, None, :] - self.ch_coords[None, :, :]
        if self.dist == "manhattan":
            distances = np.sum(np.abs(distances), axis=-1)
        elif self.dist == "euclidean":
            distances = np.linalg.norm(distances, axis=-1)
        return distances


class UtahArray(MEA):
    """Specialized class for Utah-style microelectrode arrays.

    Maps electrode channels to a 2D grid based on an array map.

    Attributes:
        arr_map: 2D grid mapping channel numbers to spatial locations.
        sep: Inter-electrode spacing in millimeters.
    """

    def __init__(
        self,
        arr_map: np.ndarray | None = None,
        locs: np.ndarray | Iterable | None = None,
        sep: float = 0.4,
        dist: str = "manhattan",
    ) -> None:
        """Initialize a UtahArray grid mapping.

        Args:
            arr_map: 2D numpy array mapping channel
                numbers to their physical grid positions. If None, uses
                the standard UTAH_MAP.
            locs: Explicit spatial coordinates. Usually
                generated automatically from `arr_map` and `sep`.
                Defaults to None.
            sep: Physical distance between electrode
                shanks in millimeters. Defaults to 0.4.
            dist: Distance metric for parent MEA class.
                Defaults to "manhattan".
        """
        if arr_map is None:
            arr_map = UTAH_MAP
        assert arr_map.ndim == 2
        self.arr_map = arr_map
        self.chs = np.unique(self.arr_map[~np.isnan(arr_map)])
        self.sep = sep
        # locs argument is for parent compatibility
        coords = locs if locs is not None else self._ch_coords()
        super().__init__(locs=coords, dist=dist)

    def _ch_coords(self) -> np.ndarray:
        """Generates 2D coordinates for each channel based on grid pitch.

        Returns:
            Array of (N, 2) coordinates.
        """
        coords = []
        for channel in self.chs:
            (
                row,
                col,
            ) = np.where(self.arr_map == channel)
            coords.append([row[0], col[0]])
        return self.sep * np.array(coords)

    def form_data(self, data: np.ndarray) -> np.ndarray:
        """Reshape a vector of channel data into the 2D array grid shape.

        Args:
            data: Data of shape (self.num_ch,).

        Returns:
            2D grid of data.
        """
        arr = np.zeros_like(self.arr_map) * np.nan
        for idx, channel in enumerate(self.chs):
            channel_idx = np.where(self.arr_map == channel)
            arr[channel_idx] = data[idx]
        return arr


class Subject:
    """Represents an experimental subject (e.g., non-human primate).

    Maintains information about the subject's brain regions, implanted
    electrode arrays, and channel-to-region mappings.

    Attributes:
        name: Subject name or identifier.
        arrays: list of MEA objects.
        regions: list of region names corresponding to arrays.
        array_map: Mapping of region names to MEA objects.
        region_chs: Mapping of region names to global channel slices.
        num_ch: Total number of channels across all arrays.
    """

    def __init__(self, name: str, arrays: list[MEA], regions: list[str]) -> None:
        """Initialize a new Subject representation.

        Args:
            name: Semantic identifier for the subject (e.g. 'U').
            arrays: list of implanted electrode array objects.
            regions: list of semantic names for the brain
                regions where arrays are implanted (e.g. ['m1', 'pmd']).
                Length must match `arrays`.
        """
        self.name = name
        self.arrays = arrays
        self.regions = regions
        self.array_map = dict(zip(self.regions, self.arrays))

        start_ix = 0
        self.region_chs: dict[str, slice] = {}
        self.num_ch = 0
        for region, arr in zip(self.regions, self.arrays):
            self.region_chs[region] = slice(start_ix, start_ix + arr.num_ch)
            self.num_ch += arr.num_ch
            start_ix += arr.num_ch

    def __repr__(self) -> str:
        """Return string representation of the Subject."""
        return (
            f"subject(name={self.name}, chs={self.num_ch}, " f"regions={self.regions})"
        )

    def ch_region(self, channel: int) -> str:
        """Identify which semantic region a channel index belongs to.

        Args:
            channel: 1-indexed channel number.

        Returns:
            Region name.

        Raises:
            ValueError: If the channel is not found in any region.
        """
        channel_ix = channel - 1
        for region in self.regions:
            if (
                self.region_chs[region].start
                <= channel_ix
                < self.region_chs[region].stop
            ):
                return region
        raise ValueError(f"channel {channel} not found")


UTAH = UtahArray()
