"""
Utilities for serializing and deserializing Python objects using cloudpickle.

This module provides wrappers for pickle operations with support for
preparing Study/Span objects for serialization by closing HDF5 handles.
"""

from __future__ import annotations

from typing import Any
import cloudpickle
from .. import abstracts


def pickle_proof(obj: Any) -> Any:
    """Prepare objects for serialization by closing open HDF5 handles.

    Args:
        obj: The object to prepare for serialization (Span, Study, or SpanSet).

    Returns:
        The same object with closed HDF5 handles.

    Raises:
        ValueError: If obj is not a Span, StudyMixin, or SpanSet.
    """
    if isinstance(obj, abstracts.Span):
        studies = [obj.study]
    elif isinstance(obj, abstracts.StudyMixin):
        studies = [obj]
    elif isinstance(obj, abstracts.SpanSet):
        studies = list({span.study for span in obj.spans})
    else:
        raise ValueError("Not a Span, Study, or SpanSet!")

    for study in studies:
        if isinstance(study, abstracts.HeadH5Study) and study.head_handle is not None:
            study.head_handle.close()
            study.head_handle = None
            study.has.discard("head")
    return obj


def pickle_down(obj: Any, file_path: str | None = None) -> bytes | None:
    """Serialize an object to a file or bytes using cloudpickle.

    Args:
        obj: The object to serialize.
        file_path: The destination file path. If None, returns bytes.

    Returns:
        Serialized bytes if file_path is None, otherwise None.
    """
    obj = pickle_proof(obj)
    if file_path is None:
        return cloudpickle.dumps(obj)

    with open(file_path, "wb") as file_handle:
        cloudpickle.dump(obj, file_handle)
    return None


def pickle_up(source: str | bytes) -> Any:
    """Deserialize an object from a file or bytes using cloudpickle.

    Args:
        source: The source file path (str) or bytes to deserialize.

    Returns:
        The deserialized Python object.
    """
    if isinstance(source, bytes):
        return cloudpickle.loads(source)

    with open(source, "rb") as file_handle:
        return cloudpickle.load(file_handle)
