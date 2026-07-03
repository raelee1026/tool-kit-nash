from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class TraceResult(dict):
    """JSON-serializable NASH trace with a convenience writer."""

    def to_json(self, path: str | Path) -> None:
        save_trace(self, path)


def save_trace(trace: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(trace), file, ensure_ascii=False, indent=2)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, TraceResult):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value
