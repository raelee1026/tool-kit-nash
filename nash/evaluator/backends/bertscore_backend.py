from __future__ import annotations

import inspect

from ..types import NumericMention


class BERTScoreBackend:
    """Token-level backend for BERTScore-style NASH scoring.

    The textual channel uses BERTScore F1 on masked sentences. Numeric
    alignment uses the underlying Hugging Face encoder: token embeddings whose
    character offsets overlap each numeric mention are averaged into one
    contextual number vector.
    """

    name = "bertscore"

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from bert_score.scorer import BERTScorer
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERTScoreBackend requires bert-score, transformers, and torch. "
                "Install them with `pip install -e '.[model]'`."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.scorer = BERTScorer(
            model_type=model_name,
            lang="en",
            rescale_with_baseline=False,
            device=self.device,
            batch_size=batch_size,
            idf=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(f"{model_name} does not expose a fast tokenizer for span offsets.")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def text_similarity(self, sentences1: list[str], sentences2: list[str]) -> list[float]:
        if len(sentences1) != len(sentences2):
            raise ValueError("sentences1 and sentences2 must have the same length.")
        _, _, f1 = self.scorer.score(sentences1, sentences2)
        if hasattr(f1, "detach"):
            return [float(value) for value in f1.detach().cpu().tolist()]
        return [float(value) for value in f1]

    def encode_sentence(self, sentence: str) -> list[float]:
        vectors = self._sentence_hidden_states(sentence)
        vector = vectors.mean(dim=0)
        return [float(value) for value in vector.detach().cpu().tolist()]

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

    def tokenization_info(
        self,
        sentence: str,
        mentions: list[NumericMention],
    ) -> tuple[list[str], list[list[int]]]:
        encoded = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        spans = [
            _overlapping_token_indices(offsets, mention.start, mention.end)
            for mention in mentions
        ]
        return tokens, [[indices[0], indices[-1] + 1] if indices else [] for indices in spans]

    def _sentence_hidden_states(self, sentence: str):
        hidden_states, _ = self._hidden_states_and_offsets(sentence)
        return hidden_states

    def _hidden_states_and_offsets(self, sentence: str):
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )
        offsets = [tuple(offset) for offset in encoded.pop("offset_mapping")[0].tolist()]
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        encoded = _filter_model_inputs(self.model, encoded)
        with self._torch.no_grad():
            outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state[0]
        real_indices = [index for index, (start, end) in enumerate(offsets) if not (start == 0 and end == 0)]
        if real_indices:
            hidden_states = hidden_states[real_indices]
            offsets = [offsets[index] for index in real_indices]
        return hidden_states, offsets


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
