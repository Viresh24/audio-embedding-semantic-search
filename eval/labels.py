"""Sentence-transformer judge: which dataset labels a search prompt is about."""

from __future__ import annotations

import numpy as np


class LabelEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        return vecs

    def label_sims(self, query: str, labels: list[str]) -> dict[str, float]:
        if not labels:
            return {}
        q = self.encode([query])[0]
        m = self.encode(labels)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = m @ q
        return {lab: float(s) for lab, s in zip(labels, sims)}

    def matching_labels(
        self,
        query: str,
        labels: list[str],
        tau: float = 0.5,
        top_m: int | None = None,
    ) -> dict[str, float]:
        sims = self.label_sims(query, labels)
        if top_m:
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:top_m]
            return {lab: s for lab, s in ranked}
        return {lab: s for lab, s in sims.items() if s >= tau}
