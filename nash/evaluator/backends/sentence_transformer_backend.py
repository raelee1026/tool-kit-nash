from __future__ import annotations

import inspect
from collections.abc import Sequence

from ..types import NumericMention


class SentenceTransformerBackend:
    """SentenceTransformer backend with upstream-style token-span alignment."""

    name = "sentence-transformer"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend. "
                "Install the package with `pip install -e .` or use a custom backend."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer, self.alignment_model = _find_sentence_transformer_backbone(self.model)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(f"{model_name} does not expose a fast tokenizer for span offsets.")
        self.device = str(self.model.device)
        self.alignment_model = self.alignment_model.to(self.device)
        self.alignment_model.eval()

    def _encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [embedding.astype(float).tolist() for embedding in embeddings]

    def text_similarity(self, sentences1: list[str], sentences2: list[str]) -> list[float]:
        if len(sentences1) != len(sentences2):
            raise ValueError("sentences1 and sentences2 must have the same length.")
        left = self._encode_many(sentences1)
        right = self._encode_many(sentences2)
        return [_dot(a, b) for a, b in zip(left, right)]

    def encode_sentence(self, sentence: str) -> list[float]:
        embeddings = self._encode_many([sentence])
        return embeddings[0]

    def number_embeddings(self, sentence: str, mentions: list[NumericMention]) -> list[list[float]]:
        if not mentions:
            return []
        hidden_states, offsets = self._hidden_states_and_offsets(sentence)
        vectors = []
        for mention in mentions:
            token_indices = _overlapping_token_indices(offsets, mention.start, mention.end)
            if token_indices:
                vector = hidden_states[token_indices].mean(dim=0)
            else:
                vector = hidden_states.mean(dim=0)
            vectors.append([float(value) for value in vector.detach().cpu().tolist()])
        return vectors

    def _hidden_states_and_offsets(self, sentence: str):
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )
        offsets = [tuple(offset) for offset in encoded.pop("offset_mapping")[0].tolist()]
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        encoded = _filter_model_inputs(self.alignment_model, encoded)
        with self._torch.no_grad():
            outputs = self.alignment_model(**encoded)
        hidden_states = outputs.last_hidden_state[0]
        return hidden_states, offsets


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _find_sentence_transformer_backbone(model: object):
    for module in model._modules.values():
        if hasattr(module, "tokenizer") and hasattr(module, "auto_model"):
            tokenizer = module.tokenizer
            auto_model = module.auto_model
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, auto_model
    raise RuntimeError("Could not locate the Transformer backbone inside the SentenceTransformer model.")


def _filter_model_inputs(model: object, encoded: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(model.forward)
    accepted = set(signature.parameters.keys())
    return {name: tensor for name, tensor in encoded.items() if name in accepted}


def _overlapping_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    indices = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_start == 0 and token_end == 0:
            continue
        if token_end <= start or token_start >= end:
            continue
        indices.append(index)
    return indices
