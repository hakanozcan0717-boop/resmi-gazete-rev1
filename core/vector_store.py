# -*- coding: utf-8 -*-
"""
Qdrant Vector Store Modülü

Bu dosya RAG için metin parçalarını embedding vektörlerine çevirir
ve Qdrant Cloud vektör veritabanına kaydeder.

Eski sistem:
    ChromaDB + vector_db klasörü

Yeni sistem:
    Qdrant Cloud

Gerekli environment variables:
    QDRANT_URL
    QDRANT_API_KEY
    QDRANT_COLLECTION
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    SentenceTransformers + Qdrant tabanlı vektör veritabanı sınıfı.
    """

    def __init__(
        self,
        persist_dir: str = "vector_db",
        collection_name: str = "resmi_gazete",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", collection_name)
        self.model_name = model_name

        if not self.qdrant_url:
            raise ValueError(
                "QDRANT_URL bulunamadı. "
                "Render Environment Variables ve GitHub Secrets içine QDRANT_URL eklemelisin."
            )

        if not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY bulunamadı. "
                "Render Environment Variables ve GitHub Secrets içine QDRANT_API_KEY eklemelisin."
            )

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60,
        )

        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Qdrant collection yoksa oluşturur.
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _stable_point_id(self, raw_id: str) -> int:
        """
        Qdrant point id için metin ID'sinden stabil integer üretir.
        """
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def add_documents(self, chunks: List[Dict], batch_size: int = 128) -> int:
        """
        Metin parçalarını Qdrant içine ekler.
        """
        if not chunks:
            return 0

        total_added = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk["text"] for chunk in batch]

            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).tolist()

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

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total_added += len(batch)
            print(f"[QDRANT] {total_added}/{len(chunks)} parça Qdrant'a eklendi/güncellendi.")

        return total_added

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Kullanıcı sorusuna en yakın metin parçalarını Qdrant'tan getirir.
        """
        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()[0]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

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
