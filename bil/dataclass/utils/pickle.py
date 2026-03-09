"""
Utilities for serializing and deserializing Python objects using cloudpickle.

This module provides wrappers for pickle operations with support for
preparing Study/Span objects for serialization by closing HDF5 handles.
"""

import cloudpickle
from .. import abstracts


def pickle_proof(obj):
    """Prepare objects for serialization by closing open HDF5 handles.

    Args:
        obj (object): The object to prepare.

    Returns:
        object: The prepared object.
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


def pickle_down(obj, file_path=None):
    """Serialize an object to a file or bytes using cloudpickle.

    Args:
        obj (object): The object to serialize.
        file_path (str or Path, optional): The destination file path.
            If None, returns the serialized bytes.

    Returns:
        bytes or None: Serialized bytes if file_path is None, else None.
    """
    obj = pickle_proof(obj)
    if file_path is None:
        return cloudpickle.dumps(obj)

    with open(file_path, "wb") as file_handle:
        return cloudpickle.dump(obj, file_handle)


def pickle_up(source):
    """Deserialize an object from a file or bytes using cloudpickle.

    Args:
        source (str, Path, or bytes): The source file path or bytes.

    Returns:
        object: The deserialized Python object.
    """
    if isinstance(source, bytes):
        return cloudpickle.loads(source)

    with open(source, "rb") as file_handle:
        return cloudpickle.load(file_handle)
