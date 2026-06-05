"""
API implementation for data format NPSL.

This module handles flat HDF5 datasets common in format NPSL recordings,
providing semantic access to neural rasters and cursor kinematics.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd

from . import abstracts
from .utils import subject

if TYPE_CHECKING:
    from .abstracts import HeadH5Study

SUBJECTS: dict[str, subject.Subject] = {
    "L": subject.Subject("L", [subject.UTAH], ["m1"]),
    "J": subject.Subject("J", [subject.UTAH, subject.UTAH], ["m1", "pmd"]),
}


class DataCatalog(abstracts.DataCatalog):
    """Catalog of data access methods for format NPSL."""

    def kinematics(self) -> np.ndarray:
        """Retrieve cursor kinematics.

        Returns:
            Kinematic data of shape (2, time).
        """
        return self._return_data("kinematics")

    def raster(self, region: str | None = None) -> np.ndarray:
        """Retrieve spike raster data.

        Args:
            region: Semantic brain region.

        Returns:
            Raster data of shape (channels, time).
        """
        return self._return_data("raster", region=region)


class Span(DataCatalog, abstracts.Span):
    """Span implementation for format NPSL."""

    study: HeadH5Study

    def _get(self, h5_dataset: h5py.Dataset) -> np.ndarray:
        """Slice and return data from an HDF5 dataset.

        Args:
            h5_dataset: The dataset to read from.

        Returns:
            Transposed sliced data.
        """
        return h5_dataset[self.start : self.stop].T

    def _get_regions(self, prefix: str, region: str | None = None) -> np.ndarray:
        """Retrieve concatenated data from multiple regions.

        Args:
            prefix: Dataset prefix (e.g., 'spikeRaster').
            region: Semantic brain region.

        Returns:
            Concatenated data from the specified region(s).

        Raises:
            ValueError: If the region is unrecognized.
        """
        if region is None:
            if self.study.subject.regions == ["m1", "pmd"]:
                suffixes = ["", "2"]
            elif self.study.subject.regions == ["m1"]:
                suffixes = [""]
            else:
                suffixes = [""]  # Default fallback
        elif region.lower() in ["m1", "pmd"]:
            assert region in self.study.subject.regions, (
                f"subject {self.study.subject.name} does not have region {region}!"
            )
            # Find index of region to determine suffix
            idx = self.study.subject.regions.index(region.lower())
            suffixes = ["" if idx == 0 else str(idx + 1)]
        else:
            raise ValueError(f"unrecognized region {region}, expected m1, pmd or None")

        output = []
        for suffix in suffixes:
            output.append(self._get(self.study.head[f"{prefix}{suffix}"]))
        output_arr = np.concatenate(output, axis=0)
        return output_arr

    def _raster(self, region: str | None = None) -> np.ndarray:
        """Internal raster retrieval.

        Args:
            region: Semantic brain region.

        Returns:
            Raster data.
        """
        return self._get_regions("spikeRaster", region)

    def _kinematics(self) -> np.ndarray:
        """Internal kinematic retrieval.

        Returns:
            XY cursor position.
        """
        pos_cursor = self._get(self.study.head["cursorPos"])
        pos_cursor_xy = pos_cursor[:2]
        return pos_cursor_xy

    def _eye(self) -> np.ndarray:
        """Internal eye position retrieval.

        Returns:
            Eye position.
        """
        pos_eye = self._get(self.study.head["eyePos"])
        return pos_eye

    def _data(self, func: str, **kwargs: Any) -> np.ndarray | None:
        """Internal data retrieval router.

        Args:
            func: Data function name.
            **kwargs: Arguments.

        Returns:
            Requested data.
        """
        if func == "raster":
            return self._raster(**kwargs)
        if func == "kinematics":
            return self._kinematics()
        return None


class SpanSet(DataCatalog, abstracts.SpanSet):
    """Collection of Span objects for format NPSL."""

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
    """Array representation of format NPSL Spans."""


class StudyBase(abstracts.HeadH5Study):
    """Base class for format NPSL studies."""

    data_paths: dict[str, list[str]] = {
        "head": ["{run}.h5"],
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
        prefix = study_id[0]
        if prefix not in SUBJECTS:
            raise ValueError(f"Cannot find subject with name {prefix}")
        self.subject = SUBJECTS[prefix]

    @cached_property
    def tlen(self) -> int:
        """Return total duration based on the 'counter' dataset length.

        Returns:
            Duration in ms.
        """
        return int(self.head["counter"].shape[0])

    def by_time(self, key: int | slice) -> Span:
        """Index by relative ms time.

        Args:
            key: Time index or slice in milliseconds.

        Returns:
            A Span object.
        """
        return abstracts.by_time_ms(self, key, Span)

    def _init_meta(self) -> None:
        """Initialize metadata by parsing the HDF5 head file."""
        if self._df is not None:
            return

        path = self.fetcher.get_file("trial.csv")
        dataframe = pd.read_csv(path)
        dataframe = dataframe.sort_values(by="number").reset_index(drop=True)
        self.df = dataframe
        return


class Study(abstracts.PublicMixin, StudyBase, SpanSet):
    """Implementation of Study object for format NPSL using HTTPS fetching."""
