from __future__ import annotations

from typing import Protocol

from ..types import NumericMention


class SimilarityBackend(Protocol):
    name: str

    def text_similarity(self, sentences1: list[str], sentences2: list[str]) -> list[float]:
        ...

    def encode_sentence(self, sentence: str) -> list[float]:
        ...

    def number_embeddings(self, sentence: str, mentions: list[NumericMention]) -> list[list[float]]:
        ...
