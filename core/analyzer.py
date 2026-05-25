# -*- coding: utf-8 -*-
import sqlite3
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple
from core.database import GazetteDB
from core.utils import now_iso, tokenize_tr
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    KMeans = None
    cosine_similarity = None

class GazetteAnalyzer:
    def __init__(self, db: GazetteDB):
        self.db = db

    def corpus_word_frequency(self, top_n: int = 50) -> List[Tuple[str, int]]: # veri tabanındaki tüm belgelerdeki kelimelerin frekansını hesaplar ve en sık geçenleri döndürür
        counter = Counter()
        for row in self.db.all_texts():
            counter.update(tokenize_tr((row["title"] or "") + " " + (row["content"] or "")))
        return counter.most_common(top_n)

    def document_keywords(self, text: str, top_n: int = 15) -> List[Tuple[str, int]]: # bir metindeki en sık geçen kelimeleri döndürür
        return Counter(tokenize_tr(text)).most_common(top_n)

    def build_report(self) -> Dict: #veri tabanindaki tüm belgeleri özetleyen genel bir analiz raporu oluşturur
        rows = self.db.all_texts()
        lengths = [len(r["content"] or "") for r in rows]
        report = {
            "created_at": now_iso(),
            "total_documents": len(rows),
            "category_distribution": self.db.stats_by_category(),
            "date_distribution": self.db.stats_by_date(),
            "institution_distribution": self.db.stats_by_institution(20),
            "top_words": self.corpus_word_frequency(50),
            "content_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": round(statistics.mean(lengths), 2) if lengths else 0,
                "median": round(statistics.median(lengths), 2) if lengths else 0,
            },
        }
        if TfidfVectorizer and KMeans and len(rows) >= 5: # belgeler yeterliyse ve scikit-learn yüklüyse, içeriklerine göre kümelendirir ve her küme için özet bilgi ekler
            report["topic_clusters"] = self.topic_clusters(rows, n_clusters=min(5, max(2, len(rows) // 3)))
        else:
            report["topic_clusters"] = "scikit-learn yok veya belge sayısı yetersiz. Kurulum: pip install scikit-learn"
        return report

    def topic_clusters(self, rows: Sequence[sqlite3.Row], n_clusters: int = 5) -> List[Dict]: #veri tabanındaki belgeleri içeriklerine göre kümelendirir ve her küme için özet bilgi döndürür
        texts = [(r["title"] or "") + "\n" + (r["content"] or "")[:12000] for r in rows]
        ids = [r["id"] for r in rows]
        titles = [r["title"] for r in rows]
        vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.9, tokenizer=tokenize_tr, token_pattern=None) # TF-IDF vektörleştirici oluşturur, Türkçe tokenizasyonu kullanır ve çok nadir veya çok yaygın kelimeleri filtreler
        X = vectorizer.fit_transform(texts) # belgeleri TF-IDF vektörlerine dönüştürür
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # KMeans modeli oluşturur, küme sayısını ve rastgele durumu belirler
        labels = model.fit_predict(X) # KMeans modelini eğitir ve her belge için küme etiketlerini alır
        terms = vectorizer.get_feature_names_out() # vektörleştiricinin kelime listesini alır
        clusters = [] # her küme için, küme etiketlerine göre belgeleri gruplar, küme merkezindeki en önemli kelimeleri belirler ve örnek belgelerle birlikte özet bilgi oluşturur
        for c in range(n_clusters): # her küme için
            idxs = [i for i, lab in enumerate(labels) if lab == c] # o küme etiketine sahip belgelerin indekslerini bulur
            center = model.cluster_centers_[c] # o küme merkezinin TF-IDF vektörünü alır
            top_term_idx = center.argsort()[-10:][::-1] 
            clusters.append({ # küme bilgisi oluşturur
                "cluster": c, 
                "size": len(idxs),
                "keywords": [terms[i] for i in top_term_idx],
                "sample_documents": [{"id": ids[i], "title": titles[i]} for i in idxs[:5]],
            })
        return clusters

    def similar_documents(self, item_id: int, limit: int = 5) -> List[Dict]: #veri tabanındaki belgeler arasında belirtilen ID'ye sahip belgeye benzer olanları bulur ve benzerlik skorlarıyla birlikte döndürür
        if not TfidfVectorizer or not cosine_similarity: 
            return [{"error": "Benzerlik analizi için scikit-learn gerekir: pip install scikit-learn"}]
        rows = self.db.all_texts() 
        if len(rows) < 2:
            return []
        index = None
        texts = []
        for i, r in enumerate(rows): # tüm belgeleri dolaşır, belirtilen ID'ye sahip belgenin indeksini bulur ve her belgenin metnini (başlık + içerik) bir listeye ekler
            if r["id"] == item_id:
                index = i
            texts.append((r["title"] or "") + "\n" + (r["content"] or "")[:12000])
        if index is None:
            return []
        vectorizer = TfidfVectorizer(max_features=3000, tokenizer=tokenize_tr, token_pattern=None)
        X = vectorizer.fit_transform(texts) # tüm belgeleri TF-IDF vektörlerine dönüştürür
        sims = cosine_similarity(X[index], X).flatten() # belirtilen belgenin vektörünü tüm belgelerin vektörleriyle karşılaştırarak benzerlik skorları elde eder
        ranked = sorted([(s, i) for i, s in enumerate(sims) if i != index], reverse=True)[:limit] # benzerlik skorlarına göre diğer belgeleri sıralar ve en benzer olanları seçer
        return [{"id": rows[i]["id"], "date": rows[i]["date"], "title": rows[i]["title"], "similarity": round(float(score), 4), "item_url": rows[i]["item_url"]} for score, i in ranked]
