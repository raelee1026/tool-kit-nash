from __future__ import annotations

from typing import Any

from .metric import NASH
from .trace import TraceResult


def infer_backend(model: str, metric: str = "nash") -> str:
    lowered = model.lower()
    if metric in {"baseline", "bertscore"} or "bert-base" in lowered or "finbert" in lowered or "deberta" in lowered:
        return "bertscore"
    return "sentence-transformer"


def score_pair(
    sentence_a: str,
    sentence_b: str,
    metric: str = "nash",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    backend: str | None = None,
    threshold: float | str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Score one sentence pair with a selected baseline model.

    ``metric='baseline'`` returns the same trace schema with ``score`` set to
    the baseline score. ``metric='nash'`` sets ``score`` to the NASH score. The
    full trace is included so NASH DEMO can visualize it.
    """

    selected_backend = backend or infer_backend(model, metric)
    trace = NASHMetric(
        baseline=model,
        backend_type=selected_backend,
        threshold=threshold,
        device=device,
    ).score(sentence_a, sentence_b)
    baseline_score = float(trace["baseline_comparison"]["baseline_score"])
    nash_score = float(trace["baseline_comparison"]["nash_score"])
    selected_score = baseline_score if metric in {"baseline", "bertscore"} else nash_score
    return {
        "sentence1": sentence_a,
        "sentence2": sentence_b,
        "metric": metric,
        "model": model,
        "backend": selected_backend,
        "score": selected_score,
        "baseline_score": baseline_score,
        "nash_score": nash_score,
        "text_score": float(trace["textual_similarity"]["score"]),
        "numeric_score": float(trace["numeric_similarity"]["score"]),
        "trace": trace,
    }


def score_pairs(
    pairs: list[dict[str, str]],
    metric: str = "nash",
    models: list[str] | None = None,
    threshold: float | str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    selected_models = models or ["sentence-transformers/all-MiniLM-L6-v2"]
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        sentence_a = pair.get("sentence1") or pair.get("sentence_a") or pair.get("sentenceA") or pair.get("s1")
        sentence_b = pair.get("sentence2") or pair.get("sentence_b") or pair.get("sentenceB") or pair.get("s2")
        if not sentence_a or not sentence_b:
            continue
        for model in selected_models:
            row = score_pair(
                sentence_a,
                sentence_b,
                metric=metric,
                model=model,
                threshold=threshold,
                device=device,
            )
            row["index"] = index
            row["id"] = pair.get("id", f"pair_{index + 1}")
            row["title"] = pair.get("title", f"Pair {index + 1}")
            if pair.get("task"):
                row["task"] = pair["task"]
            if pair.get("role"):
                row["role"] = pair["role"]
            rows.append(row)
    return rows


class NASHMetric:
    """Demo-friendly pairwise scoring wrapper.

    ``NASH`` remains the lower-level metric class. ``NASHMetric`` uses the
    teacher-facing naming of "baseline" and returns the shared visualization
    trace as a JSON-serializable ``TraceResult``.
    """

    def __init__(
        self,
        baseline: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend_type: str | Any = "sentence-transformer",
        threshold: float | str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.baseline = baseline
        self.backend_type = backend_type
        self.metric = NASH(
            model_name=baseline,
            backend=backend_type,
            threshold=threshold,
            device=device,
            batch_size=batch_size,
        )

    def score(self, sentence1: str, sentence2: str) -> TraceResult:
        trace = TraceResult(self.metric.explain(sentence1, sentence2))
        trace["metadata"] = {
            **trace.get("metadata", {}),
            "baseline": self.baseline,
            "baseline_id": self.baseline,
            "baseline_label": self.baseline,
            "backend": getattr(self.metric.backend, "name", str(self.backend_type)),
            "source": "package",
        }
        return trace
