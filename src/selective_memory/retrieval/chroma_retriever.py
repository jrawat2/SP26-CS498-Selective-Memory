from __future__ import annotations

from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "selective_memory_messages",
    ) -> None:
        self.model_name = model_name
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)

        # In-memory Chroma client for now.
        # Later you can switch to PersistentClient if needed.
        self.client = chromadb.Client()
        self.collection = None

        self.texts: List[str] = []

    def fit(self, texts: List[str]) -> None:
        self.texts = texts

        existing = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        self.collection = self.client.create_collection(name=self.collection_name)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [str(i) for i in range(len(texts))]
        metadatas = [{"doc_index": i} for i in range(len(texts))]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_index: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        if self.collection is None:
            raise ValueError("Retriever is not fitted. Call fit() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        n_results = min(len(self.texts), top_k + 1 if exclude_index is not None else top_k)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        retrieved: List[tuple[int, float]] = []
        seen = set()

        for doc_id, distance in zip(ids, distances):
            doc_index = int(doc_id)

            if exclude_index is not None and doc_index == exclude_index:
                continue
            if doc_index in seen:
                continue

            seen.add(doc_index)

            # Chroma returns distance where lower is better.
            # Convert to a similarity-like score for consistency.
            score = 1.0 - float(distance)

            retrieved.append((doc_index, score))

            if len(retrieved) == top_k:
                break

        return retrieved