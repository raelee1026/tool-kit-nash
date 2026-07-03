from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable


_TOKEN_RE = re.compile(r"\[NUM\]|\b\w+\b", flags=re.UNICODE)


def tokenize_for_idf(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def build_corpus_idf(masked_sentences: Iterable[str]) -> dict[str, float]:
    document_frequency: dict[str, int] = defaultdict(int)
    document_count = 0

    for sentence in masked_sentences:
        tokens = set(tokenize_for_idf(sentence))
        if not tokens:
            continue
        document_count += 1
        for token in tokens:
            document_frequency[token] += 1

    if document_count == 0:
        return {"[num]": 1.0}

    idf = {
        token: math.log((document_count + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in document_frequency.items()
    }
    idf.setdefault("[num]", 1.0)
    return idf


def compute_alpha(masked_sentence1: str, masked_sentence2: str, idf: dict[str, float] | None = None) -> float:
    text_mass, numeric_mass = compute_idf_masses(masked_sentence1, masked_sentence2, idf)
    if not tokenize_for_idf(masked_sentence1) and not tokenize_for_idf(masked_sentence2):
        return 0.5
    denominator = text_mass + numeric_mass
    if denominator <= 0.0:
        return 0.5
    return float(max(0.0, min(1.0, text_mass / denominator)))


def compute_idf_masses(
    masked_sentence1: str,
    masked_sentence2: str,
    idf: dict[str, float] | None = None,
) -> tuple[float, float]:
    idf = {} if idf is None else idf
    tokens = tokenize_for_idf(masked_sentence1) + tokenize_for_idf(masked_sentence2)

    text_mass = 0.0
    numeric_mass = 0.0
    for token in tokens:
        if token == "[num]":
            numeric_mass += float(idf.get(token, 1.0))
        else:
            text_mass += float(idf.get(token, 1.0))
    return text_mass, numeric_mass


def combine_scores(alpha: float, text_score: float, numeric_score: float) -> float:
    return float(alpha * text_score + (1.0 - alpha) * numeric_score)
