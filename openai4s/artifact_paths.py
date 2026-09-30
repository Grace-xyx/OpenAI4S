"""Shared visibility rules for files the Artifact capture pipeline records."""

from __future__ import annotations

import os
from pathlib import Path

JUNK_DIR_SEGMENTS = frozenset({"__pycache__", "node_modules", "site-packages", "venv"})


def ignored_artifact_path(path: Path) -> bool:
    if any(part.startswith(".") for part in path.parts):
        return True
    if any(
        part in JUNK_DIR_SEGMENTS or part.endswith((".egg-info", ".dist-info"))
        for part in path.parts
    ):
        return True
    return path.name.endswith((".pyc", ".pyo"))


def require_capturable_destination(workspace: Path, target: Path) -> None:
    """Refuse a destination the normal workspace sweep deliberately omits.

    This is a capture-visibility check, not a filesystem authorization check.
    Callers still acquire the existing pinned workspace parent for I/O.
    """
    try:
        relative = target.relative_to(workspace)
    except ValueError:
        raise ValueError("dataset destination must be inside its workspace") from None
    if not relative.parts:
        raise ValueError("dataset destination must name a file within the workspace")
    if ignored_artifact_path(relative):
        raise ValueError("dataset destination is excluded from Artifact capture")
    _require_exact_existing_spelling(workspace, relative)
    parent = target.parent
    while True:
        try:
            (parent / ".git").lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ValueError(
                "dataset destination capture visibility is unavailable"
            ) from error
        else:
            raise ValueError(
                "dataset destination is inside a repository excluded from Artifact capture"
            )
        if parent == workspace:
            return
        parent = parent.parent


def _require_exact_existing_spelling(workspace: Path, relative: Path) -> None:
    """Refuse a name that reaches an existing entry spelled differently.

    The receipt names the file as the caller spelled it, while the capture
    sweep names it as the directory stores it. On a case- or
    normalization-insensitive filesystem (macOS and Windows defaults)
    ``DATASETS/x.csv`` lands in an existing ``datasets/`` and the two never
    match, so the import would fail only after publishing the file. A name
    that does not exist yet is created with the requested spelling.
    """
    current = workspace
    for part in relative.parts:
        try:
            names = os.listdir(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ValueError(
                "dataset destination capture visibility is unavailable"
            ) from error
        if part not in names:
            try:
                os.lstat(current / part)
            except FileNotFoundError:
                return
            except OSError as error:
                raise ValueError(
                    "dataset destination capture visibility is unavailable"
                ) from error
            raise ValueError(
                "dataset destination differs from an existing workspace entry "
                "only by letter case or Unicode normalization; use its exact name"
            )
        current = current / part
