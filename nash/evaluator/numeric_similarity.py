from __future__ import annotations

from .types import NumericAlignment, NumericMention


def scale_adaptive_normalizer(v: float, u: float) -> float:
    return 1.0 + 0.5 * (abs(v) + abs(u))


def magnitude_similarity(v: float, u: float) -> float:
    normalizer = scale_adaptive_normalizer(v, u)
    return float(1.0 / (1.0 + abs(v - u) / normalizer))


def directional_numeric_score(
    source_mentions: list[NumericMention],
    target_mentions: list[NumericMention],
    alignments: list[NumericAlignment],
) -> float:
    if not source_mentions:
        return 1.0
    if not target_mentions:
        return 0.0

    score_by_source = {
        alignment.source_index: alignment.magnitude_similarity
        for alignment in alignments
        if alignment.valid
    }
    return float(sum(score_by_source.get(index, 0.0) for index in range(len(source_mentions))) / len(source_mentions))


def bidirectional_numeric_score(
    mentions1: list[NumericMention],
    mentions2: list[NumericMention],
    alignments_1_to_2: list[NumericAlignment],
    alignments_2_to_1: list[NumericAlignment],
) -> tuple[float, float, float]:
    if not mentions1 and not mentions2:
        return 1.0, 1.0, 1.0
    if not mentions1 or not mentions2:
        forward = directional_numeric_score(mentions1, mentions2, alignments_1_to_2)
        reverse = directional_numeric_score(mentions2, mentions1, alignments_2_to_1)
        return 0.0, forward, reverse

    forward = directional_numeric_score(mentions1, mentions2, alignments_1_to_2)
    reverse = directional_numeric_score(mentions2, mentions1, alignments_2_to_1)
    return float(0.5 * (forward + reverse)), forward, reverse
