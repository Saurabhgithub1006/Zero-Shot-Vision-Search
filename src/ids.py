import hashlib
import os

from src.config import PROJECT_ROOT


def to_rel_posix(path, root=PROJECT_ROOT):
    """Project-relative path with forward slashes, identical on every OS."""
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path, root)
        except ValueError:
            pass  # different drive on Windows: keep the absolute path
    return os.path.normpath(path).replace(os.sep, "/")


def image_id(path, root=PROJECT_ROOT):
    """Deterministic vector id for an image, derived from its relative path."""
    return hashlib.md5(to_rel_posix(path, root).encode()).hexdigest()
