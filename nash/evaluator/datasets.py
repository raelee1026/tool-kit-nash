from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NashDataset:
    name: str
    records: list[dict[str, Any]]
    path: str | None = None

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


PAIR_FIELD_ALIASES = {
    "sentence1": ("sentence1", "s1", "text1", "premise", "source", "sentence_a", "a", "left"),
    "sentence2": ("sentence2", "s2", "text2", "hypothesis", "target", "sentence_b", "b", "right"),
}

TRIPLET_FIELD_ALIASES = {
    "anchor": ("anchor", "query", "sentence_a", "s1"),
    "positive": ("positive", "pos", "sentence_b", "s2_positive", "positive_sentence"),
    "negative": ("negative", "neg", "sentence_c", "s2_negative", "negative_sentence"),
}

LISTWISE_FIELD_ALIASES = {
    "anchor": ("anchor", "query", "sentence_a", "s1"),
    "variants": ("variants", "candidates", "choices", "items"),
}


_DATASET_FILES = {
    "numfine_triplet": ["triplet_test.json"],
    "numfine_crosspair": ["crosspair_test.json"],
    "numfine_listwise": ["listwise_test.json"],
}


def load_dataset(name: str, data_dir: str | Path | None = None, split: str | None = None) -> NashDataset:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in _DATASET_FILES:
        choices = ", ".join(sorted(_DATASET_FILES))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {choices}.")

    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    candidate_files = _candidate_files(normalized, root, split)
    records: list[dict[str, Any]] = []
    loaded_paths: list[str] = []

    for path in candidate_files:
        if path.exists():
            records.extend(_load_records(path))
            loaded_paths.append(str(path))

    if not records:
        expected = ", ".join(path.name for path in candidate_files)
        raise FileNotFoundError(f"Could not find local data for '{name}' under {root}. Expected one of: {expected}.")

    return NashDataset(name=normalized, records=normalize_records(records), path=";".join(loaded_paths))


def normalize_records(records: list[dict[str, Any]], protocol: str | None = None) -> list[dict[str, Any]]:
    """Normalize common pair/triplet/listwise dataset schemas into NASH keys."""

    normalized = []
    for index, record in enumerate(records):
        item = _normalize_record(record, protocol)
        if item:
            item.setdefault("id", str(record.get("id", record.get("uid", index))))
            normalized.append(item)
    return normalized


def normalize_pair_record(record: dict[str, Any]) -> dict[str, Any] | None:
    sentence1 = _text_value(record, PAIR_FIELD_ALIASES["sentence1"])
    sentence2 = _text_value(record, PAIR_FIELD_ALIASES["sentence2"])
    if not sentence1 or not sentence2:
        return None
    normalized = {
        **record,
        "sentence1": sentence1,
        "sentence2": sentence2,
    }
    if "score" not in normalized:
        score = _first(record, ("label", "gold", "similarity", "target", "rating"))
        if score is not None:
            normalized["score"] = score
    return normalized


def normalize_triplet_record(record: dict[str, Any]) -> dict[str, Any] | None:
    anchor = _text_value(record, TRIPLET_FIELD_ALIASES["anchor"])
    positive = _text_value(record, TRIPLET_FIELD_ALIASES["positive"])
    negative = _text_value(record, TRIPLET_FIELD_ALIASES["negative"])
    if not (anchor and positive and negative):
        return None
    return {
        **record,
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
    }


def normalize_listwise_record(record: dict[str, Any]) -> dict[str, Any] | None:
    anchor = _text_value(record, LISTWISE_FIELD_ALIASES["anchor"])
    variants = _first(record, LISTWISE_FIELD_ALIASES["variants"])
    if not anchor or not isinstance(variants, list):
        return None

    normalized_variants = []
    for item in variants:
        if isinstance(item, str):
            normalized_variants.append({"text": item})
        elif isinstance(item, dict):
            text = _text_value(item, ("text", "sentence", "candidate", "value"))
            if text:
                normalized_variants.append({**item, "text": text})

    if not normalized_variants:
        return None
    return {
        **record,
        "anchor": anchor,
        "variants": normalized_variants,
    }


def _candidate_files(name: str, root: Path, split: str | None) -> list[Path]:
    if split:
        stemmed = []
        for filename in _DATASET_FILES[name]:
            path = Path(filename)
            stemmed.append(path.with_name(f"{path.stem}_{split}{path.suffix}"))
            stemmed.append(path.with_name(f"{split}_{path.name}"))
        return [root / filename for filename in stemmed] + [root / filename for filename in _DATASET_FILES[name]]
    return [root / filename for filename in _DATASET_FILES[name]]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "examples"):
            if isinstance(payload.get(key), list):
                return [record for record in payload[key] if isinstance(record, dict)]
    raise ValueError(f"Unsupported dataset JSON shape in {path}.")


def _normalize_record(record: dict[str, Any], protocol: str | None) -> dict[str, Any] | None:
    if protocol:
        requested = protocol.strip().lower().replace("_", "-")
        if requested == "triplet":
            return normalize_triplet_record(record)
        if requested in {"pair", "crosspair", "cross-pair", "stsb", "finsts"}:
            return normalize_pair_record(record)
        if requested == "listwise":
            return normalize_listwise_record(record)

    return normalize_triplet_record(record) or normalize_listwise_record(record) or normalize_pair_record(record)


def _text_value(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first(record, keys)
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("text", "sentence", "value"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None
