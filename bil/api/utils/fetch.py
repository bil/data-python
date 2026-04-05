"""
Unified fetching interface for the BIL data API.

This module provides abstract and concrete implementations for retrieving
scientific data from various remote endpoints (HTTPS, Rclone).
"""

from __future__ import annotations
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING


import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

if TYPE_CHECKING:
    # Circular otherwise; only need this for typing.
    from ..abstracts import HeadH5Study

# HDF5 Cache settings
H5CACHE_RDCC_NBYTES = 200 * 1024 * 1024  # 200MB
H5CACHE_RDCC_NSLOTS = 1009


class Fetcher(ABC):
    """Abstract base class for data fetchers.

    Handles semantic name resolution and basic directory management.

    Attributes:
        base_url: Remote base URL or rclone remote path.
        download_dir: Local directory path where fetched files are stored.
        quiet: If True, suppresses progress indicators.
        options: Additional fetcher-specific options.
    """

    def __init__(
        self,
        base_url: str,
        download_dir: str | Path,
        quiet: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Fetcher base class.

        Args:
            base_url: Remote base URL or rclone remote path where
                source data files are located.
            download_dir: Local directory path where fetched
                files will be stored. This directory will be created if
                it does not exist.
            quiet: If True, suppresses progress indicators
                (like tqdm bars or rclone -P output) and informational
                print statements. Defaults to False.
            **kwargs: Additional fetcher-specific options stored in `self.options`.
        """
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        self.quiet = quiet
        self.options = kwargs
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def check_file_exists(self, file_name: str) -> bool:
        """Verify file existence at source.

        Args:
            file_name: Path relative to base_url.

        Returns:
            True if file exists, False otherwise.
        """

    @abstractmethod
    def get_file(self, file_name: str) -> str:
        """Fetch a single file.

        Args:
            file_name: Path relative to base_url.

        Returns:
            Absolute path to the local file.
        """

    @abstractmethod
    def get_files(self, files: list[str]) -> list[str]:
        """Fetch multiple files.

        Args:
            files: list of paths relative to base_url.

        Returns: list of absolute paths to local files.
        """

    def get_data(self, study: HeadH5Study, *names: str) -> list[Path]:
        """Semantic data fetcher using data_paths from the study.

        Args:
            study: The study object requesting data.
            *names: Semantic names (e.g., 'lfp') or explicit paths.

        Returns:
            Local paths to the fetched files.

        Raises:
            FileNotFoundError: If any required file is missing at source.
        """
        files_fetch: list[str] = []
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
    """Fetcher for HTTP/HTTPS endpoints using requests and tqdm.

    Attributes:
        session: Persistent requests Session with retry logic.
    """

    def __init__(
        self,
        base_url: str,
        download_dir: str | Path,
        quiet: bool = False,
        **kwargs: Any,
    ) -> None:
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

    def check_file_exists(self, file_name: str) -> bool:
        """Verify file existence via HTTP HEAD request."""
        url = f"{self.base_url}/{file_name}"
        try:
            response = self.session.head(url, timeout=60_000)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_file(self, file_name: str) -> str:
        """Fetch file over HTTPS with progress bar using tempfile atomic writes."""
        output_path = self.download_dir / file_name
        output_path.parent.mkdir(exist_ok=True, parents=True)

        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        if not self.quiet:
            print(f"Copying {file_name} to {self.download_dir}...")

        url = f"{self.base_url}/{file_name}"

        try:
            # Create the temp file in the exact same directory as the target destination
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                # Store the path so we can rename/delete it later
                temp_path = Path(temp_file.name)

                with self.session.get(url, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0)) or None

                    with tqdm(
                        total=total,
                        unit="iB",
                        unit_scale=True,
                        desc=file_name,
                        disable=self.quiet,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            temp_file.write(chunk)
                            pbar.update(len(chunk))
                        # Ensure all internal buffers related to file are written to disk
                        temp_file.flush()
                        os.fsync(temp_file.fileno())

            # Atomic replacement
            if output_path.exists() and output_path.stat().st_size > 0:
                # Another process beat us (for case of concurrent downloads)
                temp_path.unlink(missing_ok=True)
            else:
                # Atomic swap
                os.replace(str(temp_path), str(output_path))

        except Exception as err:
            # Clean up the temp file if the download fails
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FileNotFoundError(f"Failed to download {url}: {err}") from err

        return str(output_path)

    def get_files(self, files: list[str]) -> list[str]:
        """Fetch multiple files sequentially.

        Args:
            files: list of file names to fetch.

        Returns: list of local paths to the fetched files.
        """
        return [self.get_file(f) for f in files]

    def get_yaml(self, study_id: str) -> dict[str, Any]:
        """Fetch and parse session YAML configuration.

        Args:
            study_id: ID of the study to fetch metadata for.

        Returns:
            Dictionary of parsed YAML data.
        """
        path = self.get_file(f"{study_id}.yaml")
        with open(path, "r", encoding="utf-8") as file_handle:
            return yaml.safe_load(file_handle)

    def __del__(self) -> None:
        """Ensure the persistent session is closed on deletion."""
        if hasattr(self, "session"):
            self.session.close()


class FetcherRclone(Fetcher):
    """Fetcher for remote storage using rclone subprocess calls.

    Attributes:
        flags: list of rclone command flags.
    """

    flags: list[str] = ["--transfers", "16", "--stats", "10s"]

    def check_file_exists(self, file_name: str) -> bool:
        """Verify file existence via rclone lsjson.

        Args:
            file_name: Name of the file to check.

        Returns:
            True if file exists, False otherwise.
        """
        cmd = ["rclone", "lsjson", f"{self.base_url}/{file_name}"]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def get_file(self, file_name: str) -> str:
        """Fetch file via rclone copyto.

        Args:
            file_name: Name of the file to fetch.

        Returns:
            Absolute path to the local file.

        Raises:
            FileNotFoundError: If rclone fails to copy the file.
        """
        if not self.quiet:
            print(f"Copying {file_name} to {self.download_dir}...")
        dest = self.download_dir / file_name
        dest.parent.mkdir(exist_ok=True, parents=True)

        cmd = [
            "rclone",
            "copyto",
            f"{self.base_url}/{file_name}",
            str(dest),
            *self.flags,
        ]
        if not self.quiet:
            cmd.append("-P")

        try:
            subprocess.run(cmd, check=True, capture_output=self.quiet)
        except Exception as err:
            raise FileNotFoundError(
                f"Rclone failed to copy {file_name}: {err}"
            ) from err

        return str(dest)

    def get_files(self, files: list[str]) -> list[str]:
        """Fetch multiple files efficiently using include patterns.

        Args:
            files: list of file names to fetch.

        Returns: list of local paths to the fetched files.

        Raises:
            subprocess.CalledProcessError: If the bulk copy fails.
        """
        if not self.quiet:
            print(f"Copying {files} to {self.download_dir}...")

        cmd = ["rclone", "copy", self.base_url, str(self.download_dir), *self.flags]
        if not self.quiet:
            cmd.append("-P")
        for file_name in files:
            cmd.extend(["--include", f"/{file_name}"])

        try:
            subprocess.run(cmd, check=True, capture_output=self.quiet)
        except subprocess.CalledProcessError as err:
            if not self.quiet:
                print(f"Rclone bulk copy failed: {err}")
            raise err

        return [str(self.download_dir / f) for f in files]
