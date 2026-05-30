# -*- coding: utf-8 -*-
"""
RAG Modülü - Genel Hybrid Filtreli Sürüm

Bu sürüm sadece vektör benzerliğiyle yetinmez.
Ayrıca:
- soru niyetini tespit eder,
- pozitif/negatif anahtar kelime filtresi uygular,
- kategori, başlık ve metni birlikte değerlendirir,
- sonuçları yeniden puanlar.

Amaç:
"kanun" sorusunda ihale ilanı gelmesini,
"atama" sorusunda taşınmaz satışı gelmesini,
"ihale" sorusunda akademik kadro ilanı gelmesini azaltmak.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

from core.database import GazetteDB
from core.vector_store import VectorStore

try:
    from core.llm_client import LLMClient
except Exception:
    LLMClient = None


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Uzun metni daha düzgün parçalara böler.
    """
    if not text:
        return []

    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    current = ""

    for line in lines:
        if len(line) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", line)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current) + len(sentence) + 1 <= chunk_size:
                    current += (" " if current else "") + sentence
                else:
                    if len(current) > 100:
                        chunks.append(current.strip())

                    tail = current[-overlap:].strip() if current else ""
                    if " " in tail:
                        tail = tail[tail.find(" ") + 1:]

                    current = (tail + " " + sentence).strip() if tail else sentence
        else:
            if len(current) + len(line) + 1 <= chunk_size:
                current += ("\n" if current else "") + line
            else:
                if len(current) > 100:
                    chunks.append(current.strip())

                tail = current[-overlap:].strip() if current else ""
                if " " in tail:
                    tail = tail[tail.find(" ") + 1:]

                current = (tail + "\n" + line).strip() if tail else line

    if len(current) > 100:
        chunks.append(current.strip())

    return chunks


class RAGEngine:
    """
    Resmî Gazete için RAG işlemlerini yöneten sınıf.
    """

    def __init__(self, db_path: str = "resmi_gazete.db", vector_db_path: str = "vector_db"):
        self.db = GazetteDB(db_path)
        self.vector_store = VectorStore(persist_dir=vector_db_path)

    def _normalize_text(self, text: str) -> str:
        """
        Türkçe metni arama ve filtreleme için basitleştirir.
        """
        text = text or ""
        text = text.lower()

        replacements = {
            "ı": "i", "İ": "i", "ğ": "g", "ü": "u", "ş": "s",
            "ö": "o", "ç": "c", "â": "a", "î": "i", "û": "u",
            "ý": "i", "þ": "s", "ð": "g", "Ý": "i", "Þ": "s", "Ð": "g",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = "".join(
            char for char in unicodedata.normalize("NFKD", text)
            if unicodedata.category(char) != "Mn"
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _query_terms(self, question: str) -> List[str]:
        """
        Soru içindeki anlamlı kelimeleri çıkarır.
        """
        q = self._normalize_text(question)
        terms = re.findall(r"[a-z0-9]+", q)

        stop = {
            "ile", "ilgili", "olan", "olanlari", "seyleri", "getir",
            "listele", "nelerdir", "nedir", "hakkinda", "bana",
            "kaynaklara", "gore", "ozetle", "resmi", "gazete",
            "var", "hangi", "nedir", "ne"
        }

        return [t for t in terms if len(t) > 2 and t not in stop]

    def _detect_intent(self, question: str) -> str:
        """
        Kullanıcının sorusunun ana niyetini tespit eder.
        """
        q = self._normalize_text(question)

        intent_rules = [
            ("atama", ["atama", "atanma", "atandi", "atanmistir", "gorevden alma", "gorevden alindi", "kadro", "personel"]),
            ("ihale", ["ihale", "artirma", "eksiltme", "sartname", "teminat", "satilacaktir", "kiralama"]),
            ("kanun", ["kanun", "kanunu", "kanunlari", "kanun no", "kabul tarihi"]),
            ("yonetmelik", ["yonetmelik", "yonetmeligi", "yonetmeliginde"]),
            ("teblig", ["teblig", "tebligi", "sira no"]),
            ("karar", ["cumhurbaskani karari", "karar sayisi", "kurul karari", "karar no"]),
            ("kamulastirma", ["kamulastirma", "kamulastirilmasina", "acele kamulastirma", "istimlak"]),
            ("universite_kadro", ["ogretim uyesi", "ogretim elemani", "profesor", "docent", "dr ogr", "universitesi", "rektorlugunden"]),
            ("vergi", ["vergi", "maliye", "gelir idaresi", "kdv", "otv", "harc"]),
            ("saglik", ["saglik", "ilac", "tibbi", "hastane", "sgk", "sosyal guvenlik"]),
        ]

        for intent, words in intent_rules:
            if any(w in q for w in words):
                return intent

        return "genel"

    def _intent_profile(self, intent: str) -> Dict[str, List[str]]:
        """
        Her niyet için pozitif/negatif kelime listesi döndürür.
        """
        profiles = {
            "atama": {
                "positive": [
                    "atama", "atanma", "atanmistir", "atandi", "gorevden alma",
                    "gorevden alindi", "goreve atan", "kadro", "personel",
                    "rektorlugunden", "universitesi", "profesor", "docent",
                    "dr ogr", "ogretim uyesi", "ogretim elemani", "akademik kadro"
                ],
                "negative": [
                    "ihale", "artirma", "eksiltme", "sartname", "teminat",
                    "satilacaktir", "kiralama", "acik artirma", "tasinmaz satilacaktir",
                    "tasinmaz", "ada parsel", "ada/parsel"
                ],
                "categories": ["atama", "karar", "diger"],
            },
            "ihale": {
                "positive": [
                    "ihale", "artirma", "eksiltme", "sartname", "teminat",
                    "satilacaktir", "kiralama", "malzeme satin alinacak",
                    "yapim isi", "acik artirma", "kapali teklif", "pazarlik usulu"
                ],
                "negative": [
                    "atama", "atanma", "gorevden alma", "profesor",
                    "docent", "ogretim uyesi", "ogretim elemani"
                ],
                "categories": ["ihale"],
            },
            "kanun": {
                "positive": [
                    "kanun no", "kanun numarasi", "kanunu", "kanun",
                    "kabul tarihi", "turkiye buyuk millet meclisi",
                    "tbmm", "kanunun", "madde 1"
                ],
                "negative": [
                    "ihale sartnamesi", "sartname", "teminat", "satilacaktir",
                    "acik artirma", "tasinmaz satilacaktir", "vakfin adi",
                    "vakfedenler", "vakfin amaci", "ilan olunur",
                    "kamulastirilmasina karar", "ada parsel", "ada/parsel"
                ],
                "categories": ["kanun"],
            },
            "yonetmelik": {
                "positive": [
                    "yonetmelik", "yonetmeligi", "yonetmeliginde",
                    "degisiklik yapilmasina dair yonetmelik",
                    "madde", "yururluk", "yurutur"
                ],
                "negative": [
                    "ihale", "sartname", "satilacaktir", "teminat",
                    "vakfin adi", "ilan olunur"
                ],
                "categories": ["yonetmelik"],
            },
            "teblig": {
                "positive": [
                    "teblig", "tebligi", "sira no", "genel tebligi",
                    "resmi gazete", "madde", "yururluk"
                ],
                "negative": [
                    "ihale", "sartname", "satilacaktir", "teminat",
                    "vakfin adi", "ilan olunur"
                ],
                "categories": ["teblig"],
            },
            "karar": {
                "positive": [
                    "karar", "karari", "karar sayisi", "karar no",
                    "cumhurbaskani karari", "kurul karari", "karar tarihi"
                ],
                "negative": [
                    "ihale sartnamesi", "sartname", "satilacaktir",
                    "acik artirma", "vakfin adi", "ilan olunur"
                ],
                "categories": ["karar"],
            },
            "kamulastirma": {
                "positive": [
                    "kamulastirma", "kamulastirilmasina", "acele kamulastirma",
                    "istimlak", "irtifak", "parsel", "ada/parsel", "kamu yarari"
                ],
                "negative": [
                    "ogretim uyesi", "profesor", "docent", "ihale sartnamesi"
                ],
                "categories": ["karar", "diger"],
            },
            "universite_kadro": {
                "positive": [
                    "universitesi", "rektorlugunden", "ogretim uyesi",
                    "ogretim elemani", "profesor", "docent", "dr ogr",
                    "fakulte", "bolum", "anabilim dali", "kadro unvani"
                ],
                "negative": [
                    "ihale", "sartname", "teminat", "satilacaktir",
                    "ada parsel", "vakfin adi"
                ],
                "categories": ["atama", "diger", "ilan"],
            },
            "vergi": {
                "positive": [
                    "vergi", "maliye", "gelir idaresi", "kdv", "otv",
                    "harc", "beyanname", "matrah", "tahakkuk"
                ],
                "negative": [
                    "ihale", "sartname", "satilacaktir", "ogretim uyesi"
                ],
                "categories": ["vergi/maliye", "teblig", "kanun", "karar"],
            },
            "saglik": {
                "positive": [
                    "saglik", "ilac", "tibbi", "hastane", "sgk",
                    "sosyal guvenlik", "tedavi", "eczane"
                ],
                "negative": [
                    "ihale sartnamesi", "satilacaktir", "ada parsel"
                ],
                "categories": ["saglik", "teblig", "yonetmelik", "karar"],
            },
        }

        return profiles.get(intent, {"positive": [], "negative": [], "categories": []})

    def build_index(
        self,
        limit: Optional[int] = None,
        chunk_size: int = 1200,
        overlap: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        SQLite veritabanındaki belgeleri RAG indeksine ekler.
        """
        rows = self.db.all_texts(limit=limit, start_date=start_date, end_date=end_date)
        all_chunks = []

        for row in rows:
            gazette_id = row["id"]
            title = row["title"] or ""
            content = row["content"] or ""
            full_text = title + "\n\n" + content

            chunks = split_text_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)

            for index, chunk_text in enumerate(chunks):
                chunk_id = f"gazette_{gazette_id}_chunk_{index}"

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "gazette_id": gazette_id,
                        "chunk_index": index,
                        "title": title,
                        "date": row["date"] or "",
                        "category": row["category"] or "",
                        "item_url": row["item_url"] or "",
                    }
                })

        return self.vector_store.add_documents(all_chunks)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Kullanıcının sorusuna en yakın Resmî Gazete parçalarını getirir.
        """
        return self.vector_store.search(question, top_k=top_k)

    def _is_listing_request(self, question: str) -> bool:
        q = self._normalize_text(question)
        listing_words = [
            "getir", "listele", "goster", "sirala", "bul",
            "hangi", "nelerdir", "kaynaklari", "kaynaklar"
        ]
        if any(word in q for word in listing_words):
            return True

        bare_listing_terms = [
            "kanunlari", "yonetmelikleri", "tebligleri", "kararlari",
            "ihaleleri", "atama kararlari", "atamalar", "atamalari"
        ]
        return any(term in q for term in bare_listing_terms)

    def _meaningful_question_terms(self, question: str, intent: str = "genel") -> List[str]:
        q = self._normalize_text(question)
        terms = re.findall(r"[a-z0-9]+", q)
        profile_terms = set(self._listing_profile(intent).get("terms", []))
        stop_terms = {
            "getir", "listele", "goster", "sirala", "bul", "hangi",
            "nelerdir", "kaynaklari", "kaynaklar", "ilgili", "hakkinda",
            "resmi", "gazete", "kararlari", "kararlar", "kanunlari",
            "yonetmelikleri", "tebligleri", "ihaleleri", "atamalari",
            "atamalar",
        }
        output = []
        for term in terms:
            if len(term) <= 2 or term in stop_terms or term in profile_terms:
                continue
            if term not in output:
                output.append(term)
        return output

    def _is_weak_title(self, title: str) -> bool:
        normalized = self._normalize_text(title)
        return (
            bool(re.fullmatch(r"\d{1,2}\s+\w+\s+\d{4}\s+\w+", normalized))
            or bool(re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}\s+resmi gazete pdf", normalized))
            or normalized.endswith("resmi gazete pdf")
        )

    def _listing_profiles(self) -> Dict[str, Dict]:
        return {
            "yonetmelik": {
                "terms": ["yonetmelik", "yonetmeligi", "yonetmeliginde", "yonetmeligine"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Yönetmeli(?:ği|k|ğinin|ğinde|ğine)[^.!?\n]{0,80})"
                ],
                "generic_starts": [
                    "(1) bu yonetmelikte", "bu yonetmelikte", "ilgili yonetmelikte",
                    "yonetmelikte hukum", "yonetmelikte adi gecen", "bu yonetmelikte hukum",
                ],
            },
            "kanun": {
                "terms": ["kanun", "kanunu", "kanun no", "kabul tarihi"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Kanunu[^.!?\n]{0,80})",
                    r"(Kanun No\.\s*:\s*\d+[^.!?\n]{0,160})",
                ],
                "generic_starts": [
                    "bu kanunda", "bu kanunun", "ilgili kanunda", "kanunda hukum",
                    "kanunun", "kanun uyarinca", "sayili kanunun", "bu kanunun ek",
                ],
            },
            "teblig": {
                "terms": ["teblig", "tebligi", "sira no", "genel tebligi"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Tebliği[^.!?\n]{0,80})",
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Tebliğ[^.!?\n]{0,80})",
                ],
                "generic_starts": [
                    "bu tebligde", "ilgili tebligde", "tebligde hukum", "tebligin",
                ],
            },
            "karar": {
                "terms": ["karar", "karari", "karar sayisi", "karar no"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Kararı[^.!?\n]{0,80})",
                    r"(Karar Sayısı\s*:\s*\d+[^.!?\n]{0,160})",
                ],
                "generic_starts": [
                    "bu kararda", "ilgili kararda", "kararda hukum", "kararin",
                ],
            },
            "ihale": {
                "terms": ["ihale", "artirma", "eksiltme", "sartname"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}İhale[^.!?\n]{0,80})",
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Artırma[^.!?\n]{0,80})",
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Eksiltme[^.!?\n]{0,80})",
                ],
                "generic_starts": [
                    "ihale edilecektir", "ihale olunur", "sartname",
                ],
            },
            "atama": {
                "terms": ["atama", "atanma", "atanmistir", "gorevden alma", "kadro"],
                "patterns": [
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Atama[^.!?\n]{0,80})",
                    r"([A-ZÇĞİÖŞÜ0-9][^.!?\n]{10,220}Atanmıştır[^.!?\n]{0,80})",
                ],
                "generic_starts": [
                    "atanmistir", "atama yapilmistir", "bu atama",
                ],
            },
        }

    def _listing_profile(self, intent: str) -> Dict:
        return self._listing_profiles().get(intent, {"terms": [], "patterns": [], "generic_starts": []})

    def _is_generic_listing_title(self, title: str, intent: str = "genel") -> bool:
        normalized = self._normalize_text(title).strip(" .,:;-")
        generic_starts = self._listing_profile(intent).get("generic_starts", [])
        return (
            not normalized
            or self._is_weak_title(title)
            or any(normalized.startswith(prefix) for prefix in generic_starts)
            or bool(re.match(r"^\d+\s+sayili\s+kanunun\b", normalized))
            or bool(re.match(r"^bu\s+kanunun\b", normalized))
        )

    def _clean_extracted_title(self, title: str) -> str:
        title = re.split(r"\s+MADDE\s+\d+", title, maxsplit=1)[0]
        title = re.split(r"\s+Yürürlük\s*$", title, maxsplit=1)[0]
        return title.strip(" -–\t,;:")

    def _extract_better_title(self, title: str, text: str, intent: str = "genel") -> str:
        if title and not self._is_weak_title(title):
            return title

        profile = self._listing_profile(intent)
        terms = profile.get("terms", [])

        for raw_line in (text or "").splitlines():
            line = self._clean_extracted_title(raw_line.strip(" -–\t"))
            normalized = self._normalize_text(line)
            if len(line) < 20 or len(line) > 250:
                continue
            if any(term in normalized for term in terms) and not self._is_generic_listing_title(line, intent):
                return line

        compact_text = re.sub(r"\s+", " ", text or "")
        for pattern in profile.get("patterns", []):
            for match in re.finditer(pattern, compact_text):
                candidate = self._clean_extracted_title(match.group(1))
                if 20 <= len(candidate) <= 250 and not self._is_generic_listing_title(candidate, intent):
                    return candidate

        return title or "-"

    def _matches_intent(self, intent: str, item: Dict) -> bool:
        if intent == "genel":
            return True

        metadata = item.get("metadata", {})
        haystack = self._normalize_text(
            " ".join([
                str(metadata.get("title", "")),
                str(metadata.get("category", "")),
                str(item.get("text", "")),
            ])
        )

        terms = self._listing_profile(intent).get("terms", [])
        if not terms:
            return True

        return any(term in haystack for term in terms)

    def _source_text_for_filter(self, item: Dict) -> str:
        metadata = item.get("metadata", {})
        return self._normalize_text(
            " ".join([
                str(metadata.get("title", "")),
                str(metadata.get("category", "")),
                str(item.get("text", "")),
            ])
        )

    def _question_term_score(self, question_terms: List[str], item: Dict) -> int:
        if not question_terms:
            return 0

        haystack = self._source_text_for_filter(item)
        return sum(1 for term in question_terms if term in haystack)

    def _passes_question_terms(self, question_terms: List[str], item: Dict) -> bool:
        if not question_terms:
            return True

        # For focused queries such as "MEB atama kararları", at least one
        # non-document-type term must appear in the candidate source.
        return self._question_term_score(question_terms, item) > 0

    def _clean_source_item(self, item: Dict, intent: str) -> Optional[Dict]:
        metadata = item.get("metadata", {})
        title = self._extract_better_title(metadata.get("title", "") or "", item.get("text", ""), intent)
        date = metadata.get("date", "-") or "-"
        category = metadata.get("category", "-") or "-"
        url = metadata.get("item_url", "-") or "-"
        snippet = self._source_snippet(item.get("text", ""))

        normalized_title = self._normalize_text(title)
        normalized_category = self._normalize_text(category)
        terms = self._listing_profile(intent).get("terms", [])
        if terms and not any(term in normalized_category or term in normalized_title for term in terms):
            return None
        if terms and self._is_generic_listing_title(title, intent):
            return None

        cleaned = dict(item)
        cleaned["metadata"] = {
            **metadata,
            "title": title,
            "date": date,
            "category": category,
            "item_url": url,
            "snippet": snippet,
        }
        return cleaned

    def _source_snippet(self, text: str, limit: int = 420) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        text = re.sub(r"^(MADDE\s+\d+[-–.]?\s*)", "", text, flags=re.I).strip()
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut.rstrip(".,;:") + "..."

    def prepare_sources(self, question: str, top_k: int = 5) -> List[Dict]:
        intent = self._detect_intent(question)
        question_terms = self._meaningful_question_terms(question, intent)

        raw_limit = max(top_k * 8, 40)
        raw_results = self.retrieve(question, top_k=raw_limit)
        reranked = self._rerank_and_filter_results(question, raw_results, top_k=raw_limit)

        filtered_results = [
            item for item in reranked
            if self._matches_intent(intent, item) and self._passes_question_terms(question_terms, item)
        ]
        filtered_results.sort(
            key=lambda item: (
                -self._question_term_score(question_terms, item),
                item.get("distance", 999),
            )
        )

        seen = set()
        prepared = []
        for item in filtered_results:
            cleaned = self._clean_source_item(item, intent)
            if not cleaned:
                continue
            metadata = cleaned["metadata"]
            title = metadata.get("title", "")
            date = metadata.get("date", "")
            url = metadata.get("item_url", "")
            key = self._normalize_text(title) or url or date

            if key in seen:
                continue
            seen.add(key)
            prepared.append(cleaned)

            if len(prepared) >= top_k:
                break

        return prepared

    def _format_source_list(self, question: str, results: List[Dict]) -> str:
        lines = [f"Soru: {question}", ""]

        if not results:
            return "\n".join(lines + ["Uygun kaynak bulunamadı."])

        lines.append("Bulunan kaynaklar:")
        lines.append("")

        for item_no, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            title = metadata.get("title", "-") or "-"
            date = metadata.get("date", "-") or "-"
            category = metadata.get("category", "-") or "-"
            url = metadata.get("item_url", "-") or "-"
            lines.append(f"{item_no}. {title}")
            lines.append(f"   Tarih: {date}")
            lines.append(f"   Kategori: {category}")
            lines.append(f"   Kaynak: {url}")
            lines.append("")

        return "\n".join(lines).strip()

    def _rerank_and_filter_results(self, question: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Vektör sonuçlarını genel hybrid mantıkla filtreler ve tekrar sıralar.
        """
        intent = self._detect_intent(question)
        profile = self._intent_profile(intent)
        query_terms = self._query_terms(question)

        positive_words = profile["positive"]
        negative_words = profile["negative"]
        allowed_categories = profile["categories"]

        reranked = []

        for item in results:
            meta = item.get("metadata", {})

            raw_title = str(meta.get("title", ""))
            raw_category = str(meta.get("category", ""))
            raw_text = item.get("text", "")

            title = self._normalize_text(raw_title)
            category = self._normalize_text(raw_category)
            text = self._normalize_text(raw_text)

            full_text = f"{title} {category} {text}"

            positive_score = sum(1 for word in positive_words if word in full_text)
            negative_score = sum(1 for word in negative_words if word in full_text)
            query_score = sum(1 for term in query_terms if term in full_text)
            title_score = sum(2 for term in query_terms if term in title)

            category_score = 0
            if allowed_categories:
                for cat in allowed_categories:
                    if cat in category:
                        category_score += 3

            distance = float(item.get("distance", 999))

            if intent == "kanun":
                looks_like_real_law = (
                    "kanun" in category
                    or " kanunu" in title
                    or title.endswith("kanunu")
                    or "kanun no" in text
                    or "kabul tarihi" in text
                    or "turkiye buyuk millet meclisi" in text
                    or "tbmm" in text
                )

                if not looks_like_real_law:
                    continue

                if negative_score > 0 and "kanun" not in category and "kanunu" not in title:
                    continue

            elif intent in ["atama", "ihale", "yonetmelik", "teblig", "karar", "universite_kadro"]:
                if negative_score > 0 and positive_score == 0 and category_score == 0:
                    continue

                if positive_score == 0 and query_score == 0 and category_score == 0:
                    continue

            elif intent != "genel":
                if negative_score > 0 and positive_score == 0:
                    continue

            score = (
                positive_score * 10
                + category_score * 12
                + query_score * 6
                + title_score * 8
                - negative_score * 12
                - distance
            )

            item["hybrid_score"] = round(score, 4)
            item["intent"] = intent
            item["keyword_score"] = positive_score
            item["query_score"] = query_score
            item["category_score"] = category_score
            item["negative_score"] = negative_score

            reranked.append(item)

        if not reranked and intent == "genel":
            return results[:top_k]

        if not reranked:
            return []

        reranked.sort(key=lambda x: (-x.get("hybrid_score", 0), x.get("distance", 999)))
        return reranked[:top_k]

    def answer_without_llm(self, question: str, top_k: int = 5) -> str:
        """
        LLM bağlamadan kaynaklı cevap taslağı üretir.
        """
        is_listing = self._is_listing_request(question)
        result_limit = max(top_k, 10) if is_listing else top_k
        results = self.prepare_sources(question, top_k=result_limit)

        if is_listing:
            return self._format_source_list(question, results)

        lines = []
        lines.append(f"Soru: {question}")
        lines.append("")
        lines.append("Bu soruyla ilgili bulunan Resmî Gazete parçaları:")
        lines.append("")

        if not results:
            lines.append(
                "Uygun kaynak parçası bulunamadı. "
                "Sorguyu daha açık yazmayı deneyebilirsin. "
                "Örnek: 'Atama kararları hangi kurumlarla ilgili?', "
                "'Yayımlanan kanun metinlerini getir', "
                "'İhale ilanlarını getir'."
            )
            return "\n".join(lines)

        for i, item in enumerate(results, start=1):
            metadata = item["metadata"]

            lines.append("=" * 80)
            lines.append(f"{i}) Tarih: {metadata.get('date', '-')}")
            lines.append(f"Başlık: {metadata.get('title', '-')}")
            lines.append(f"Kategori: {metadata.get('category', '-')}")
            lines.append(f"Kaynak: {metadata.get('item_url', '-')}")
            lines.append(f"Benzerlik uzaklığı: {item.get('distance')}")
            lines.append(f"Hybrid skor: {item.get('hybrid_score')}")
            lines.append(f"Niyet: {item.get('intent')}")
            lines.append("")
            lines.append(item["text"][:1400])
            lines.append("")

        return "\n".join(lines)

    def build_prompt_for_llm(self, question: str, top_k: int = 5, sources: Optional[List[Dict]] = None) -> str:
        """
        OpenAI, Gemini, Ollama vb. bir LLM'e verilecek prompt üretir.
        """
        results = sources if sources is not None else self.prepare_sources(question, top_k=top_k)

        context_parts = []

        for i, item in enumerate(results, start=1):
            metadata = item["metadata"]

            context_parts.append(
                f"""
[KAYNAK {i}]
Tarih: {metadata.get('date', '-')}
Başlık: {metadata.get('title', '-')}
Kategori: {metadata.get('category', '-')}
URL: {metadata.get('item_url', '-')}
Hybrid skor: {item.get('hybrid_score')}

Metin:
{item["text"]}
"""
            )

        context = "\n".join(context_parts)
        if not context_parts:
            context = "İlgili temiz kaynak bulunamadı."

        prompt = f"""
DETAY SEVIYESI:
- Cevap orta detayli olsun; sadece tek cumlelik veya yuzeysel ozet verme.
- Her kaynakta soruyla ilgili hangi duzenleme, karar, ilan veya hukmun yer aldigini ayri ayri belirt.
- Kaynaklar hakkinda verilen bilgiyi eskisine gore biraz daha detayli acikla; ancak ayri "Genel ozet" veya "Sonuc" bolumu yazma.
- Ayni konuyu tekrarlayan kaynaklari birlestir; farkli tarih veya farkli hukum varsa ayri belirt.
- Kaynaklarda olmayan bilgi, yorum, tahmin veya genel bilgi ekleme.

Sen bir Resmî Gazete analiz asistanısın.

Görevin:
- Aşağıdaki temizlenmiş kaynaklardan kısa ve anlaşılır bir özet çıkar.
- Her önemli bulguyu ilgili kaynak başlığı, tarih veya kaynak numarasıyla ilişkilendir.
- Kaynaklarda olmayan bilgi, yorum, tahmin veya genel bilgi ekleme.
- Kaynaklar kullanıcının sorusuna cevap vermiyorsa sadece "Uygun kaynak bulunamadı." de.
- Kaynak varsa asla "Bu bilgi verilen kaynaklarda bulunamadı." cümlesini ekleme.
- Cevabın sonunda "Kullanılan kaynaklar" başlığıyla yararlandığın kaynakları listele.
- Cevabı Türkçe, maddeli ve kısa tut.

SORU:
{question}

KAYNAKLAR:
{context}

CEVAP:
"""

        return prompt.strip()

    def answer_with_llm(self, question: str, top_k: int = 5, model: str = None, sources: Optional[List[Dict]] = None) -> str:
        """
        RAG + LLM cevabı üretir.
        """
        sources = sources if sources is not None else self.prepare_sources(question, top_k=top_k)
        if not sources:
            return "Uygun kaynak bulunamadı."

        if LLMClient is None:
            raise RuntimeError("LLMClient yüklenemedi. core/llm_client.py dosyasını ve openai paketini kontrol et.")

        prompt = self.build_prompt_for_llm(question=question, top_k=top_k, sources=sources)
        llm = LLMClient(model=model)
        return llm.generate_answer(prompt)
