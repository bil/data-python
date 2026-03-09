"""
BIL data API package.
"""

from bil.dataclass.abstracts import REMOTE_HTTPS_URL

from . import dataclassA
from .utils import fetch

IMPLEMENTATIONS = {
    "A": dataclassA.Study,
}


class Study:
    """Factory class for creating Study instances.

    Reads the session-level YAML metadata to automatically return the correct
    implementation for a given study ID.
    """

    def __new__(cls, study_id, download_dir="./data", quiet=False):
        """Create and return a format-specific Study instance.

        Args:
            study_id (str): The unique identifier for the study session.
            download_dir (str): Local directory for data caching.
            quiet (bool, optional): If True, suppresses progress output.

        Returns:
            StudyMixin: An instance of a concrete Study implementation.

        Raises:
            FileNotFoundError: If the study cannot be found or metadata is missing.
        """
        try:
            # Discovery: fetch the YAML metadata to find the format
            metadata = fetch.FetcherHTTPS(
                f"{REMOTE_HTTPS_URL}/{study_id}", download_dir, quiet=quiet
            ).get_yaml(study_id)
            fmt = metadata.get("format")

            if fmt in IMPLEMENTATIONS:
                return IMPLEMENTATIONS[fmt](
                    study_id, download_dir=download_dir, quiet=quiet
                )

            raise ValueError(f"Unsupported data format '{fmt}' in metadata.")

        except Exception as error:
            raise FileNotFoundError(
                f"Study ID '{study_id}' could not be initialized. "
                f"Ensure that it is accessible at the remote endpoint {REMOTE_HTTPS_URL}."
                f"Original error: {str(error)}"
            ) from error
