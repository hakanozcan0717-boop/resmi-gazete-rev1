# -*- coding: utf-8 -*-
"""
Render Free icin hafif Qdrant Vector Store.

Bu surum Render icinde sentence-transformers / torch yuklemez.
Embedding islemini OpenAI Embeddings API ile uzaktan yapar.
Qdrant vektorleri saklar ve arama yapar.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "vector_db",
        collection_name: str = "resmi_gazete",
        model_name: str = "text-embedding-3-small",
    ):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", collection_name)

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", model_name)

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL bulunamadı.")
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY bulunamadı.")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY bulunamadı. OpenAI embedding için gerekli.")

        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60,
        )

        self.openai = OpenAI(api_key=self.openai_api_key)

        # text-embedding-3-small vektor boyutu
        self.vector_size = 1536

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            return

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _stable_point_id(self, raw_id: str) -> int:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def add_documents(self, chunks: List[Dict], batch_size: int = 64) -> int:
        if not chunks:
            return 0

        total_added = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk["text"] for chunk in batch]
            embeddings = self._embed_texts(texts)

            points = []

            for chunk, embedding in zip(batch, embeddings):
                metadata = chunk.get("metadata", {}) or {}

                payload = {
                    "original_id": chunk["id"],
                    "text": chunk["text"],
                    "gazette_id": metadata.get("gazette_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "title": metadata.get("title", ""),
                    "date": metadata.get("date", ""),
                    "category": metadata.get("category", ""),
                    "item_url": metadata.get("item_url", ""),
                }

                points.append(
                    models.PointStruct(
                        id=self._stable_point_id(chunk["id"]),
                        vector=embedding,
                        payload=payload,
                    )
                )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total_added += len(batch)
            print(f"[QDRANT] {total_added}/{len(chunks)} parça Qdrant'a eklendi/güncellendi.")

        return total_added

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query.strip():
            return []

        query_embedding = self._embed_texts([query])[0]

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        results = response.points

        output = []

        for hit in results:
            payload = hit.payload or {}

            output.append({
                "id": payload.get("original_id", str(hit.id)),
                "text": payload.get("text", ""),
                "metadata": {
                    "gazette_id": payload.get("gazette_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "title": payload.get("title", ""),
                    "date": payload.get("date", ""),
                    "category": payload.get("category", ""),
                    "item_url": payload.get("item_url", ""),
                },
                "distance": 1 - float(hit.score),
                "score": float(hit.score),
            })

        return output
