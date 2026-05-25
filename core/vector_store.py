# -*- coding: utf-8 -*-
"""
Vector Store Modülü

Bu dosya RAG için metin parçalarını embedding vektörlerine çevirir
ve ChromaDB isimli vektör veritabanına kaydeder.

Görevleri:
1. SentenceTransformer modeliyle metni sayısal vektöre çevirmek.
2. ChromaDB içinde bu vektörleri kalıcı olarak saklamak.
3. Kullanıcı sorusuna en yakın metin parçalarını bulmak.
"""

from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    ChromaDB + SentenceTransformers tabanlı vektör veritabanı sınıfı.
    """

    def __init__(
        self,
        persist_dir: str = "vector_db",
        collection_name: str = "resmi_gazete",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        persist_dir:
            Vektör veritabanının kaydedileceği klasör.
            Proje içinde otomatik olarak vector_db/ oluşur.

        collection_name:
            ChromaDB içindeki koleksiyon adı.

        model_name:
            Metni vektöre çevirecek embedding modeli.
            Türkçe desteklediği için multilingual model seçildi.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name

        # ChromaDB veritabanını kalıcı klasör olarak başlatır.
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Koleksiyon yoksa oluşturur, varsa mevcut olanı kullanır.
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        # Metinleri embedding vektörlerine çevirecek model.
        self.model = SentenceTransformer(model_name)

    def add_documents(self, chunks, batch_size=500):
        """
        Metin parçalarını ChromaDB içine ekler.

        ChromaDB çok büyük veriyi tek seferde kabul etmeyebilir.
        Bu yüzden kayıtları küçük paketler halinde gönderiyoruz.

        chunks:
            RAG için hazırlanmış metin parçaları listesi.

        batch_size:
            Her seferde kaç parça ChromaDB'ye gönderilecek?
        """

        if not chunks:
            return 0

        total_added = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            ids = []
            texts = []
            metadatas = []

            for chunk in batch:
                ids.append(chunk["id"])
                texts.append(chunk["text"])
                metadatas.append(chunk["metadata"])

            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            total_added += len(batch)

            print(f"[RAG] {total_added}/{len(chunks)} parça vektör veritabanına eklendi.")

        return total_added

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Kullanıcı sorusuna en yakın metin parçalarını getirir.

        query:
            Kullanıcının doğal dilde yazdığı soru.

        top_k:
            Kaç tane en yakın sonuç getirilecek?
        """
        if not query.strip():
            return []

        # Soruyu da aynı embedding modelinden geçiriyoruz.
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        output = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            output.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": distance,
            })

        return output
