"""
BIL data API.
"""

from __future__ import annotations

from typing import Any
from . import abstracts
from . import formatS
from . import formatNPSL
from .utils import fetch

IMPLEMENTATIONS: dict[str, type[formatS.Study] | type[formatNPSL.Study]] = {
    "S": formatS.Study,
    "NPSL": formatNPSL.Study,
}


def get(
    study_id: str,
    download_dir: str = "./data",
    quiet: bool = False,
    deposition_version: int | None = None,
) -> formatS.Study | formatNPSL.Study:
    """Create and return a format-specific Study instance.

    Args:
        study_id: The unique identifier for the study session.
        download_dir: Local directory for data caching.
        quiet: If True, suppress progress messages.
        deposition_version: Optional version number for the SDR data.

    Returns:
        A Study object (e.g., formatS.Study).

    Raises:
        FileNotFoundError: If the study cannot be found or metadata is missing.
        ValueError: If the data format in metadata is unsupported.
    """
    url = abstracts.sdr_url(deposition_version)
    try:
        # Discovery: fetch the YAML metadata to find the format
        metadata = fetch.FetcherHTTPS(
            f"{url}/{study_id}", f"{download_dir}/{study_id}", quiet=quiet
        ).get_yaml(study_id)
        study_format = metadata.get("format")

        if study_format in IMPLEMENTATIONS:
            return IMPLEMENTATIONS[study_format](
                study_id, download_dir=download_dir, quiet=quiet
            )

        raise ValueError(f"Unsupported data format '{study_format}' in metadata.")

    except Exception as error:
        raise FileNotFoundError(
            f"Study ID '{study_id}' could not be initialized. "
            f"Ensure that it is accessible at the remote endpoint {url}."
            f"Original error: {str(error)}"
        ) from error
