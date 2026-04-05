"""Checkpoint helpers used by the demo package."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _path(path: str | Path) -> Path:
    return Path(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: str | Path, label: str | None = None) -> Path:
    path = _path(path)
    _ensure_parent(path)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def load_pickle(path: str | Path, label: str | None = None) -> Any:
    with _path(path).open("rb") as handle:
        return pickle.load(handle)


def save_json(obj: Any, path: str | Path, label: str | None = None) -> Path:
    path = _path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=True)
    return path


def load_json(path: str | Path, label: str | None = None) -> Any:
    with _path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_exists(path: str | Path) -> bool:
    return _path(path).exists()


class CheckpointStore:
    """Small convenience wrapper around the checkpoint helper functions."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.base_dir.joinpath(*parts)

    def save_pickle(self, obj: Any, *parts: str) -> Path:
        return save_pickle(obj, self.path(*parts))

    def load_pickle(self, *parts: str) -> Any:
        return load_pickle(self.path(*parts))

    def save_json(self, obj: Any, *parts: str) -> Path:
        return save_json(obj, self.path(*parts))

    def load_json(self, *parts: str) -> Any:
        return load_json(self.path(*parts))

    def save_text(self, text: str, *parts: str) -> Path:
        path = self.path(*parts)
        _ensure_parent(path)
        path.write_text(text, encoding="utf-8")
        return path
