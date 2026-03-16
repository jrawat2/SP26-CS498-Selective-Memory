from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    query_id: int
    retrieved_msg_id: int
    rank: int
    score: float


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.texts: List[str] = []

    def fit(self, texts: List[str]) -> None:
        self.texts = texts
        self.doc_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_index: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        if self.doc_embeddings is None:
            raise ValueError("Retriever is not fitted. Call fit() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = np.dot(self.doc_embeddings, query_embedding)

        if exclude_index is not None and 0 <= exclude_index < len(scores):
            scores[exclude_index] = -np.inf

        ranked_indices = np.argsort(scores)[::-1]

        results: List[tuple[int, float]] = []
        seen = set()

        for idx in ranked_indices:
            if not np.isfinite(scores[idx]):
                continue
            if int(idx) in seen:
                continue

            seen.add(int(idx))
            results.append((int(idx), float(scores[idx])))

            if len(results) == top_k:
                break

        return results