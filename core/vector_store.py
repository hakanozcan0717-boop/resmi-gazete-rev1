# -*- coding: utf-8 -*-
"""
Render Free icin hafif Qdrant Vector Store.

Bu surum Render icinde sentence-transformers / torch yuklemez.
Embedding islemini OpenAI Embeddings API ile uzaktan yapar.
Qdrant vektorleri saklar ve arama yapar.
"""

from __future__ import annotations

import hashlib
import datetime as dt
import os
from collections import Counter
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

        if self.collection_name not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="date",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "already has" not in message:
                print(f"[QDRANT] date payload index kontrolü atlandı: {exc}")

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

    def date_counts(self, batch_size: int = 256) -> List[Dict]:
        counts = Counter()
        next_offset = None

        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=["date"],
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                date = payload.get("date") or "-"
                counts[str(date)] += 1

            if next_offset is None:
                break

        return [
            {"date": date, "chunk_count": count}
            for date, count in sorted(counts.items(), reverse=True)
        ]

    def delete_year(self, year: int) -> Dict:
        start_date = f"{year:04d}-01-01"
        end_date = f"{year + 1:04d}-01-01"
        result = self.delete_date_range(start_date, end_date)
        result["year"] = year
        return result

    def delete_date_range(self, start_date: str, end_date: str) -> Dict:
        self._ensure_payload_indexes()
        dates = self._date_values(start_date, end_date)
        batch_count = 0

        for start in range(0, len(dates), 32):
            batch = dates[start:start + 32]
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="date",
                        match=models.MatchAny(any=batch),
                    )
                ]
            )

            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
                wait=True,
            )
            batch_count += 1

        return {
            "start_date": start_date,
            "end_date": end_date,
            "date_value_count": len(dates),
            "delete_batch_count": batch_count,
            "deleted_by_filter": True,
        }

    def _date_values(self, start_date: str, end_date: str) -> List[str]:
        start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        values = []
        current = start
        while current < end:
            values.append(current.isoformat())
            current += dt.timedelta(days=1)
        return values
