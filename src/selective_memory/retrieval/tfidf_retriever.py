from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RetrievalResult:
    query_id: int
    retrieved_msg_id: int
    rank: int
    score: float


class TfidfRetriever:
    def __init__(self, stop_words: str = "english") -> None:
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.doc_matrix = None
        self.texts: List[str] = []

    def fit(self, texts: List[str]) -> None:
        self.texts = texts
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_index: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        if self.doc_matrix is None:
            raise ValueError("Retriever is not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()

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

    def retrieve_all(self, texts: List[str], top_k: int = 5) -> List[RetrievalResult]:
        self.fit(texts)
        all_results: List[RetrievalResult] = []

        for query_id, query_text in enumerate(texts):
            retrieved = self.retrieve(
                query=query_text,
                top_k=top_k,
                exclude_index=query_id,
            )

            for rank, (msg_id, score) in enumerate(retrieved, start=1):
                all_results.append(
                    RetrievalResult(
                        query_id=query_id,
                        retrieved_msg_id=msg_id,
                        rank=rank,
                        score=score,
                    )
                )

        return all_results