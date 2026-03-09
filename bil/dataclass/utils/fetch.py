"""
Unified fetching interface for the BIL data API.

This module provides abstract and concrete implementations for retrieving
scientific data from various remote endpoints (HTTPS).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# HDF5 Cache settings
H5CACHE_RDCC_NBYTES = 200 * 1024 * 1024  # 200MB
H5CACHE_RDCC_NSLOTS = 1009


class Fetcher(ABC):
    """Abstract base class for data fetchers.

    Handles semantic name resolution and basic directory management.
    """

    def __init__(self, base_url, download_dir, quiet=False, **kwargs):
        """Initialize the Fetcher base class.

        Args:
            base_url (str): Remote base URL where source data files are located.
            download_dir (str or Path): Local directory path where fetched
                files will be stored. This directory will be created if
                it does not exist.
            quiet (bool, optional): If True, suppresses progress indicators
                (like tqdm bars) and informational print statements.
                Defaults to False.
            **kwargs: Additional fetcher-specific options stored in `self.options`.
        """
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        self.quiet = quiet
        self.options = kwargs
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def check_file_exists(self, file_name):
        """Verify file existence at source.

        Args:
            file_name (str): Path relative to base_url.

        Returns:
            bool: True if file exists.
        """

    @abstractmethod
    def get_file(self, file_name):
        """Fetch a single file.

        Args:
            file_name (str): Path relative to base_url.

        Returns:
            str: Absolute path to the local file.
        """

    @abstractmethod
    def get_files(self, files):
        """Fetch multiple files.

        Args:
            files (list[str]): List of paths relative to base_url.

        Returns:
            list[str]: Absolute paths to local files.
        """

    def get_data(self, study, *names):
        """Semantic data fetcher using data_paths from the study.

        Args:
            study (StudyMixin): The study object requesting data.
            *names: Semantic names (e.g., 'lfp') or explicit paths.

        Returns:
            list[Path]: Local paths to the fetched files.

        Raises:
            FileNotFoundError: If any required file is missing at source.
        """
        files_fetch = []
        for name in names:
            if hasattr(study, "has") and name in study.has:
                continue

            if name in study.data_paths:
                files = study.data_paths[name]
                if isinstance(files, str):
                    files = [files]
                files_fetch.extend(f.format(run=study.study_id) for f in files)
            else:
                files_fetch.append(name)

        if files_fetch:
            self.get_files(files_fetch)

            if hasattr(study, "has"):
                for name in names:
                    study.has.add(name)
        output_paths = [self.download_dir / file for file in files_fetch]
        return output_paths


class FetcherHTTPS(Fetcher):
    """Fetcher for HTTP/HTTPS endpoints using requests and tqdm."""

    def __init__(self, base_url, download_dir, quiet=False, **kwargs):
        """Initialize the HTTPS fetcher with a persistent, retrying session."""
        super().__init__(base_url, download_dir, quiet=quiet, **kwargs)

        # Configure retry strategy
        retry_strategy = Retry(
            total=10,  # Total number of retries
            backoff_factor=1,  # Wait 1s, 2s, 4s, 8s, 16s between retries
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],  # Retry on these HTTP status codes
            allowed_methods=["HEAD", "GET"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if "headers" in self.options:
            self.session.headers.update(self.options)

    def check_file_exists(self, file_name):
        """Verify file existence via HTTP HEAD request."""
        url = f"{self.base_url}/{file_name}"
        try:
            r = self.session.head(url, timeout=30)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def get_file(self, file_name):
        """Fetch file over HTTPS with progress bar."""
        if not self.quiet:
            print(f"Copying {file_name} to {self.download_dir}...")
        output_path = self.download_dir / file_name
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if output_path.exists():
            if not self.quiet:
                print("Found local file!")
            return str(output_path)

        url = f"{self.base_url}/{file_name}"

        with self.session.get(url, stream=True, timeout=600) as r:
            try:
                r.raise_for_status()
            except Exception as e:
                raise FileNotFoundError(f"Failed to download {url}: {e}") from e

            total = int(r.headers.get("content-length", 0))
            if total == 0:
                total = None

            with open(output_path, "wb") as f, tqdm(
                total=total,
                unit="iB",
                unit_scale=True,
                desc=file_name,
                disable=self.quiet,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return str(output_path)

    def get_files(self, files):
        """Fetch multiple files sequentially."""
        return [self.get_file(f) for f in files]

    def get_yaml(self, study_id):
        """Fetch and parse session YAML configuration."""
        path = self.get_file(f"{study_id}.yaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __del__(self):
        """Ensure the persistent session is closed on deletion."""
        if hasattr(self, "session"):
            self.session.close()
