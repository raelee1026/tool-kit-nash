from __future__ import annotations

from typing import Any

from .datasets import normalize_records


def evaluate(dataset: Any, metric: Any, protocol: str) -> dict[str, Any]:
    records = normalize_records(list(getattr(dataset, "records", dataset)), protocol)
    normalized = protocol.strip().lower().replace("_", "-")
    if normalized == "triplet":
        return _evaluate_triplet(records, metric)
    if normalized in {"crosspair", "cross-pair"}:
        return _evaluate_crosspair(records, metric)
    if normalized == "listwise":
        return _evaluate_listwise(records, metric)
    raise ValueError("protocol must be one of: triplet, crosspair, listwise.")


def _evaluate_triplet(records: list[dict[str, Any]], metric: Any) -> dict[str, Any]:
    rows = []
    for record in records:
        anchor = _text(record, "anchor", "sentence_a")
        positive = _text(record, "positive", "sentence_b")
        negative = _text(record, "negative", "sentence_c")
        if not (anchor and positive and negative):
            continue
        pos = metric.score(anchor, positive)
        neg = metric.score(anchor, negative)
        rows.append(
            {
                "correct": pos["baseline_comparison"]["nash_score"] > neg["baseline_comparison"]["nash_score"],
                "baseline_correct": pos["baseline_comparison"]["baseline_score"] > neg["baseline_comparison"]["baseline_score"],
                "positive_nash_score": pos["baseline_comparison"]["nash_score"],
                "negative_nash_score": neg["baseline_comparison"]["nash_score"],
            }
        )
    return _accuracy_summary("triplet", rows)


def _evaluate_crosspair(records: list[dict[str, Any]], metric: Any) -> dict[str, Any]:
    rows = []
    for record in records:
        a1 = _text(record, "sentence_a1", "sentence_a")
        a2 = _text(record, "sentence_a2", "sentence_b")
        b1 = _text(record, "sentence_b1", "sentence_c")
        b2 = _text(record, "sentence_b2", "sentence_d")
        if not (a1 and a2 and b1 and b2):
            continue
        pair_a = metric.score(a1, a2)
        pair_b = metric.score(b1, b2)
        gold = str(record.get("gold", "")).lower()
        if gold in {"a", "pair_a", "1"}:
            correct = pair_a["baseline_comparison"]["nash_score"] > pair_b["baseline_comparison"]["nash_score"]
        elif gold in {"b", "pair_b", "2"}:
            correct = pair_b["baseline_comparison"]["nash_score"] > pair_a["baseline_comparison"]["nash_score"]
        else:
            continue
        rows.append({"correct": correct})
    return _accuracy_summary("crosspair", rows)


def _evaluate_listwise(records: list[dict[str, Any]], metric: Any) -> dict[str, Any]:
    rows = []
    for record in records:
        anchor = _text(record, "anchor", "sentence_a")
        variants = record.get("variants") or record.get("candidates")
        if not anchor or not isinstance(variants, list):
            continue
        variant_texts = [item if isinstance(item, str) else item.get("text") for item in variants if isinstance(item, (str, dict))]
        if len(variant_texts) < 2:
            continue
        scores = [metric.score(anchor, text)["baseline_comparison"]["nash_score"] for text in variant_texts]
        gold_order = record.get("gold_order")
        if isinstance(gold_order, list) and len(gold_order) == len(scores):
            predicted = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            rows.append({"exact_order": predicted == [int(index) for index in gold_order]})
    exact = sum(1 for row in rows if row["exact_order"])
    return {"protocol": "listwise", "n_records": len(rows), "exact_order_accuracy": exact / len(rows) if rows else 0.0}


def _accuracy_summary(protocol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in rows if row.get("correct"))
    baseline_rows = [row for row in rows if "baseline_correct" in row]
    summary = {
        "protocol": protocol,
        "n_records": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
    }
    if baseline_rows:
        baseline_correct = sum(1 for row in baseline_rows if row.get("baseline_correct"))
        summary["baseline_accuracy"] = baseline_correct / len(baseline_rows)
    return summary


def _text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value["text"]
    return None
