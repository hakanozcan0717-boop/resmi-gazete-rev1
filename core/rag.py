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
from core.utils import clean_extracted_text
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
            "var", "hangi", "nedir", "ne", "kurum", "kuruma",
            "kurumlara", "kim", "yapilan", "yapılan"
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

    def is_structured_answer_request(self, question: str) -> bool:
        return self._is_listing_request(question) or self._is_appointment_assignment_request(question)

    def _is_appointment_assignment_request(self, question: str) -> bool:
        q = self._normalize_text(question)
        if "atama" not in q and "atan" not in q:
            return False
        detail_words = [
            "kurum", "kuruma", "kurumlara", "kim", "kime", "hangi",
            "gorev", "goreve", "liste", "listele", "goster"
        ]
        return any(word in q for word in detail_words)

    def _meaningful_question_terms(self, question: str, intent: str = "genel") -> List[str]:
        q = self._normalize_text(question)
        terms = re.findall(r"[a-z0-9]+", q)
        profile_terms = set(self._listing_profile(intent).get("terms", []))
        stop_terms = {
            "getir", "listele", "goster", "sirala", "bul", "hangi",
            "nelerdir", "kaynaklari", "kaynaklar", "ilgili", "hakkinda",
            "ozet", "ozetle", "ozetler", "ozetleyin", "acikla", "aciklayin",
            "anlat", "anlatin", "bilgi", "ver", "nedir", "neler",
            "resmi", "gazete", "kararlari", "kararlar", "kanunlari",
            "yonetmelikleri", "tebligleri", "ihaleleri", "atamalari",
            "atamalar", "kurum", "kuruma", "kurumlara", "kim",
            "yapilan", "yapılan",
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
        title = clean_extracted_text(title)
        title = re.split(r"\s+MADDE\s+\d+", title, maxsplit=1)[0]
        title = re.split(r"\s+Yürürlük\s*$", title, maxsplit=1)[0]
        return title.strip(" -–\t,;:")

    def _extract_better_title(self, title: str, text: str, intent: str = "genel") -> str:
        title = clean_extracted_text(title)
        text = clean_extracted_text(text)
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
        text = clean_extracted_text(item.get("text", ""))
        title = self._extract_better_title(metadata.get("title", "") or "", text, intent)
        date = metadata.get("date", "-") or "-"
        category = clean_extracted_text(metadata.get("category", "-") or "-")
        url = metadata.get("item_url", "-") or "-"
        snippet = self._source_snippet(text)

        normalized_title = self._normalize_text(title)
        normalized_category = self._normalize_text(category)
        terms = self._listing_profile(intent).get("terms", [])
        if terms and not any(term in normalized_category or term in normalized_title for term in terms):
            return None
        if terms and self._is_generic_listing_title(title, intent):
            return None

        cleaned = dict(item)
        cleaned["text"] = text
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
        text = clean_extracted_text(text)
        text = re.sub(r"\s+", " ", text or "").strip()
        text = re.sub(r"^(MADDE\s+\d+[-–.]?\s*)", "", text, flags=re.I).strip()
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return cut.rstrip(".,;:") + "..."

    def prepare_sources(self, question: str, top_k: int = 5) -> List[Dict]:
        intent = self._detect_intent(question)
        question_terms = self._meaningful_question_terms(question, intent)
        is_assignment_request = self._is_appointment_assignment_request(question)

        raw_limit = max(top_k * 8, 40)
        search_query = self._appointment_search_query(question) if is_assignment_request else question
        raw_results = self.retrieve(search_query, top_k=raw_limit)
        reranked = self._rerank_and_filter_results(question, raw_results, top_k=raw_limit)

        filtered_results = [
            item for item in reranked
            if self._matches_intent(intent, item) and self._passes_question_terms(question_terms, item)
        ]
        if not filtered_results and question_terms:
            filtered_results = [
                item for item in reranked
                if self._matches_intent(intent, item)
            ]
        if is_assignment_request:
            real_assignment_results = [
                item for item in filtered_results
                if self._is_real_appointment_decision_source(item)
            ]
            if real_assignment_results:
                filtered_results = real_assignment_results
            else:
                title_sources = self._db_appointment_decision_sources(top_k=raw_limit)
                if title_sources:
                    filtered_results = title_sources
        filtered_results.sort(
            key=lambda item: (
                -self._appointment_source_score(item) if is_assignment_request else 0,
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

    def _db_appointment_decision_sources(self, top_k: int = 20) -> List[Dict]:
        try:
            rows = self.db._fetchall(
                """
                SELECT id, date, title, category, content, item_url
                FROM gazette_items
                WHERE title LIKE ? OR title LIKE ? OR content LIKE ?
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (
                    "%Atamalar Hakkında Kararlar%",
                    "%Atamalar Hakkinda Kararlar%",
                    "%Atamalar Hakkında Kararlar%",
                    int(top_k),
                ),
            )
        except Exception:
            return []

        output = []
        for row in rows:
            output.append({
                "id": f"gazette_{row['id']}_appointment_title",
                "text": row["content"] or row["title"] or "",
                "metadata": {
                    "gazette_id": row["id"],
                    "chunk_index": 0,
                    "title": row["title"] or "",
                    "date": row["date"] or "",
                    "category": row["category"] or "Atama",
                    "item_url": row["item_url"] or "",
                },
                "distance": 0,
                "score": 1,
                "hybrid_score": 999,
                "intent": "atama",
            })
        return output

    def _appointment_search_query(self, question: str) -> str:
        return (
            f"{question} Cumhurbaşkanı Tarafından Yapılan Atamalar Hakkında Kararlar "
            "Karar atanmıştır görevine atanmıştır"
        )

    def _is_real_appointment_decision_source(self, item: Dict) -> bool:
        haystack = self._source_text_for_filter(item)
        title = self._normalize_text(str(item.get("metadata", {}).get("title", "")))
        category = self._normalize_text(str(item.get("metadata", {}).get("category", "")))

        if "atanmistir" not in haystack:
            return False

        title_or_category = f"{title} {category}"
        if "cumhurbaskani tarafindan yapilan atamalar" in title_or_category:
            return True
        if "atamalar hakkinda karar" in title_or_category:
            return True
        return bool(re.search(r"\bkarar\s*:?\s*\d{4}\s*/\s*\d+\b", haystack))

    def _appointment_source_score(self, item: Dict) -> int:
        haystack = self._source_text_for_filter(item)
        score = 0
        for phrase, weight in [
            ("cumhurbaskani tarafindan yapilan atamalar", 50),
            ("atamalar hakkinda karar", 40),
            ("atanmistir", 30),
            ("karar:", 20),
            ("gorevine", 10),
        ]:
            if phrase in haystack:
                score += weight
        score += len(re.findall(r"\bkarar\s*:?\s*\d{4}\s*/\s*\d+\b", haystack)) * 15
        return score

    def _is_appointment_decision_title(self, item: Dict) -> bool:
        metadata = item.get("metadata", {}) or {}
        title = self._normalize_text(str(metadata.get("title", "")))
        return (
            "atamalar hakkinda karar" in title
            or "atamalar hakkinda kararlar" in title
            or "tarafindan yapilan atamalar" in title
        )

    def _looks_unreadable_appointment_text(self, item: Dict) -> bool:
        text = self._appointment_text_for_extraction(item)
        cleaned = clean_extracted_text(text or "")
        if not cleaned:
            return True

        normalized = self._normalize_text(cleaned)
        if "atanmistir" in normalized:
            return False

        noisy = sum(1 for char in cleaned if ord(char) < 32 or char in "#$%&*+<=>?@[\\]^_`{|}~\ufffd\u25a1")
        alpha = sum(1 for char in cleaned if char.isalpha())
        return alpha < 80 or noisy / max(len(cleaned), 1) > 0.18

    def _format_source_list(self, question: str, results: List[Dict]) -> str:
        lines = [f"Soru: {question}", ""]

        if not results:
            return "\n".join(lines + ["Uygun kaynak bulunamadı."])

        if self._is_appointment_assignment_request(question):
            return self._format_appointment_assignments(question, results)

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

    def _extract_appointment_assignments(self, results: List[Dict]) -> List[Dict]:
        rows = []
        seen = set()

        for item in results:
            metadata = item.get("metadata", {})
            text = self._appointment_text_for_extraction(item)
            fallback_decision = self._decision_number(metadata.get("title", "") + " " + text)

            for sentence in self._appointment_sentences(text):
                decision = self._decision_number(sentence) if self._decision_number(sentence) != "-" else fallback_decision
                for person, position in self._parse_appointment_rows(sentence):
                    if self._looks_like_bad_assignment(person, position):
                        continue

                    key = (metadata.get("date", ""), decision, person.lower(), position.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "date": metadata.get("date", "-") or "-",
                        "decision": decision,
                        "person": person,
                        "position": position,
                        "institution": self._institution_from_position(position),
                        "source": metadata.get("item_url", "-") or "-",
                    })

        return rows

    def _appointment_text_for_extraction(self, item: Dict) -> str:
        metadata = item.get("metadata", {}) or {}
        gazette_id = metadata.get("gazette_id")
        if gazette_id:
            try:
                row = self.db.get_item(int(gazette_id))
                if row:
                    title = row["title"] or ""
                    content = row["content"] or ""
                    full_text = clean_extracted_text(f"{title}\n{content}")
                    if len(full_text) > len(clean_extracted_text(item.get("text", ""))):
                        return full_text
            except Exception:
                pass

        return clean_extracted_text(item.get("text", ""))

    def _appointment_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text or "")
        candidates = re.split(r"(?<=[.!?])\s+|(?=Karar:\s*\d{4}/\d+)", text)
        return [part.strip() for part in candidates if "atanmıştır" in part.lower()]

    def _parse_appointment_sentence(self, sentence: str) -> Optional[tuple[str, str]]:
        before = re.split(r"\batanmıştır\b", sentence, maxsplit=1, flags=re.I)[0]
        before = re.sub(r"^.*?Karar\s*:\s*\d{4}/\d+\s*", "", before, flags=re.I)
        before = re.sub(r"^.*?Cumhurbaşkanlığından\s*:?\s*", "", before, flags=re.I)
        before = self._clean_assignment_value(before)
        if not before:
            return None

        words = before.split()
        if len(words) < 3:
            return None

        person_words = []
        for word in reversed(words):
            clean = word.strip(" ,;:.()[]")
            if not clean:
                continue
            normalized = self._normalize_text(clean)
            if normalized in {"dr", "prof", "doc", "av", "muh"}:
                person_words.append(clean)
                continue
            if clean[:1].isupper() and not clean.isupper() and not clean.endswith(("lığına", "liğine", "lüğüne", "sine", "ına", "ine", "una", "üne")):
                person_words.append(clean)
                if len(person_words) >= 5:
                    break
                continue
            break

        person_words = list(reversed(person_words))
        if len(person_words) < 2:
            return None

        person = self._clean_assignment_value(" ".join(person_words))
        position = self._clean_assignment_value(" ".join(words[: -len(person_words)]))
        position = re.sub(r"^(?:açık bulunan|boş bulunan)\s+", "", position, flags=re.I)
        if not position:
            return None
        return person, position

    def _appointment_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text or "")
        candidates = re.split(r"(?<=[.!?])\s+|(?=Karar:\s*\d{4}/\d+)", text)
        return [part.strip() for part in candidates if "atanmistir" in self._normalize_text(part)]

    def _parse_appointment_sentence(self, sentence: str) -> Optional[tuple[str, str]]:
        before = re.split(r"\b(?:atanmıştır|atanmistir|atanmÄ±ÅŸtÄ±r)\b", sentence, maxsplit=1, flags=re.I)[0]
        before = re.sub(r"^.*?Karar\s*:\s*\d{4}/\d+\s*", "", before, flags=re.I)
        before = re.sub(r"^.*?(?:Cumhurbaşkanlığından|CumhurbaÅŸkanlÄ±ÄŸÄ±ndan)\s*:?\s*", "", before, flags=re.I)
        before = self._clean_assignment_value(before)
        if not before:
            return None

        words = before.split()
        if len(words) < 3:
            return None

        person_words = []
        for word in reversed(words):
            clean = word.strip(" ,;:.()[]")
            if not clean:
                continue
            normalized = self._normalize_text(clean)
            if normalized in {"dr", "prof", "doc", "av", "muh"}:
                person_words.append(clean)
                continue
            if clean[:1].isupper() and not clean.isupper() and not self._looks_like_position_suffix(clean):
                person_words.append(clean)
                if len(person_words) >= 5:
                    break
                continue
            break

        person_words = list(reversed(person_words))
        if len(person_words) < 2:
            return None

        person = self._clean_assignment_value(" ".join(person_words))
        position = self._clean_assignment_value(" ".join(words[: -len(person_words)]))
        position = re.sub(r"^(?:açık bulunan|aÃ§Ä±k bulunan|boş bulunan|boÅŸ bulunan)\s+", "", position, flags=re.I)
        if not position:
            return None
        return person, position

    def _parse_appointment_rows(self, sentence: str) -> List[tuple[str, str]]:
        rows = self._parse_board_appointment_rows(sentence)
        if rows:
            return rows

        parsed = self._parse_single_appointment_row(sentence)
        if parsed:
            return [parsed]

        parsed = self._parse_appointment_sentence(sentence)
        return [parsed] if parsed else []

    def _parse_board_appointment_rows(self, sentence: str) -> List[tuple[str, str]]:
        before = re.split(r"\b(?:atanmıştır|atanmistir|atanmÄ±ÅŸtÄ±r)\b", sentence, maxsplit=1, flags=re.I)[0]
        before = re.sub(r"^.*?Karar\s*:\s*\d{4}/\d+\s*", "", before, flags=re.I)
        before = re.sub(r"^.*?(?:Cumhurbaşkanlığından|CumhurbaÅŸkanlÄ±ÄŸÄ±ndan)\s*:?\s*", "", before, flags=re.I).strip()
        if ";" not in before or not re.search(r"[–-]\s+", before):
            return []

        institution = self._clean_assignment_value(before.split(";", 1)[0])
        rows = []
        bullet_text = before.split(";", 1)[1]
        for raw_line in re.split(r"\s+[–-]\s+", " " + bullet_text):
            line = raw_line.strip()
            if not line:
                continue
            line = re.split(r",\s*\d{1,2}/\d{1,2}/\d{4}|\b\d+\s+sayılı\b|\b\d+\s+sayili\b", line, maxsplit=1, flags=re.I)[0]
            line = line.strip(" ,;")
            if not line:
                continue

            if "," in line:
                position_part, person_part = line.split(",", 1)
            else:
                match = re.match(r"(.+?(?:na|ne|n[ae]|lığına|liğine|luğuna|lüğüne|üyeliğine))\s+(.+)$", line, re.I)
                if not match:
                    continue
                position_part, person_part = match.group(1), match.group(2)

            person = self._extract_person_from_tail(person_part)
            if not person:
                continue

            position = self._clean_assignment_value(f"{institution} {position_part}")
            rows.append((person, position))
        return rows

    def _parse_single_appointment_row(self, sentence: str) -> Optional[tuple[str, str]]:
        before = re.split(r"\b(?:atanmıştır|atanmistir|atanmÄ±ÅŸtÄ±r)\b", sentence, maxsplit=1, flags=re.I)[0]
        before = re.sub(r"^.*?Karar\s*:\s*\d{4}/\d+\s*", "", before, flags=re.I)
        before = re.sub(r"^.*?(?:Cumhurbaşkanlığından|CumhurbaÅŸkanlÄ±ÄŸÄ±ndan)\s*:?\s*", "", before, flags=re.I)
        before = self._clean_assignment_value(before)
        if not before:
            return None

        position_match = re.search(
            r"(.+?(?:Başkanlığına|Bakanlığına|Yardımcılığına|Müdürlüğüne|Üyeliğine|Kuruluna|Kurumuna|Dairesine|lığına|liğine|luğuna|lüğüne))\s*,",
            before,
            re.I,
        )
        if not position_match:
            return None

        position = self._clean_assignment_value(position_match.group(1))
        position = re.sub(r"^.*\b(?:boşalan|bosalan)\s+", "", position, flags=re.I)
        position = re.sub(r"^.*\b(?:açık bulunan|acik bulunan)\s+", "", position, flags=re.I)
        tail = before[position_match.end():]
        parts = re.split(r"\bgereğince\b|\bgeregince\b", tail, flags=re.I)
        person_tail = parts[-1] if parts else tail
        person = self._extract_person_from_tail(person_tail)
        if not person:
            return None
        return person, position

    def _extract_person_from_tail(self, text: str) -> str:
        text = self._clean_assignment_value(text)
        text = re.sub(r"\b(?:atanmasına|atanmasina|atanması|atanmasi)\b.*$", "", text, flags=re.I)
        words = [word.strip(" ,;:.()[]") for word in text.split()]
        person_words = []
        for word in reversed(words):
            if not word:
                continue
            normalized = self._normalize_text(word)
            if normalized in {"ve", "ile", "geregince"}:
                break
            if self._looks_like_role_word(normalized) and person_words:
                break
            if self._looks_like_person_word(word):
                person_words.append(word)
                if len(person_words) >= 5:
                    break
                continue
            if person_words:
                break

        person_words = list(reversed(person_words))
        if len(person_words) < 2:
            return ""
        return self._clean_assignment_value(" ".join(person_words))

    def _looks_like_person_word(self, word: str) -> bool:
        if not any(char.isalpha() for char in word):
            return False
        normalized = self._normalize_text(word)
        if self._looks_like_role_word(normalized):
            return False
        return word[:1].isupper() or word.isupper()

    def _looks_like_role_word(self, normalized: str) -> bool:
        role_words = {
            "baskan", "baskani", "baskanligi", "yardimcisi", "yardimcisi",
            "genel", "mudur", "muduru", "kaymakami", "bakan", "uyesi",
            "ikinci", "kurul", "kurulu", "dairesi", "idaresi",
        }
        return normalized in role_words or normalized.endswith(("ligina", "ligine", "lugune", "lugune", "cisi", "muduru", "kaymakami"))

    def _looks_like_position_suffix(self, word: str) -> bool:
        normalized = self._normalize_text(word)
        return normalized.endswith(("ligina", "ligine", "lugune", "sine", "ina", "ine", "una", "une"))

    def _clean_assignment_value(self, value: str) -> str:
        value = clean_extracted_text(value or "")
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" -–—,;:.")
        return value

    def _looks_like_bad_assignment(self, person: str, position: str) -> bool:
        joined = f"{person} {position}"
        if any(char in joined for char in "\ufffd\u25a1"):
            return True
        if len(person.split()) > 8 or len(position) < 8:
            return True
        normalized_person = self._normalize_text(person)
        bad_person_terms = {"karar", "resmi", "gazete", "cumhurbaskani", "madde"}
        return any(term in normalized_person for term in bad_person_terms)

    def _institution_from_position(self, position: str) -> str:
        value = re.sub(
            r"\b(?:açık bulunan|bos bulunan|boş bulunan|görevine|kadrosuna)\b",
            " ",
            position,
            flags=re.I,
        )
        value = re.sub(r"\s+", " ", value).strip(" ,")
        for suffix in [" Bakanlığı", " Başkanlığı", " Genel Müdürlüğü", " Müdürlüğü", " Üniversitesi", " Valiliği"]:
            idx = value.find(suffix)
            if idx >= 0:
                return value[:idx + len(suffix)].strip(" ,")
        return value

    def _decision_number(self, text: str) -> str:
        match = re.search(r"Karar\s*:\s*(\d{4}/\d+)", text or "", re.I)
        return match.group(1) if match else "-"

    def _build_appointment_extraction_prompt(self, question: str, sources: List[Dict]) -> str:
        context_parts = []
        for i, item in enumerate(sources[:8], start=1):
            metadata = item.get("metadata", {}) or {}
            text = self._appointment_text_for_extraction(item)
            text = re.sub(r"\s+", " ", text or "").strip()
            if len(text) > 12000:
                text = text[:12000].rsplit(" ", 1)[0] + "..."

            context_parts.append(
                f"""
[KAYNAK {i}]
Tarih: {metadata.get('date', '-')}
Baslik: {metadata.get('title', '-')}
Kategori: {metadata.get('category', '-')}
URL: {metadata.get('item_url', '-')}

Metin:
{text}
"""
            )

        context = "\n".join(context_parts).strip() or "Ilgili kaynak metni yok."
        return f"""
Sen Resmi Gazete atama kararlarindan kisi ve kurum/gorev tablosu cikaran bir asistansin.

Gorev:
- Sadece asagidaki kaynak metinlerinde acikca gorunen atama bilgilerini kullan.
- Cevabi markdown tablo olarak ver: Tarih | Karar | Kisi | Atandigi kurum/gorev | Kaynak.
- "atanmistir", "atanmasina karar verilmistir", "gorevine atanmistir" ve benzeri ifadeleri atama olarak kabul et.
- Bir kararda birden fazla kisi varsa her kisiyi ayri satir yaz.
- Kisi adini ve kurum/gorevi tahmin etme; kaynakta acik degilse satir yazma.
- Kaynak metinleri ilgili ama hic kisi/kurum cifti ayrismiyorsa sadece "Kaynakta ayristirilabilir kisi/kurum bilgisi yok." de.
- Kaynaklar soruya cevap vermiyorsa sadece "Uygun kaynak bulunamadi." de.
- Cevabi Turkce ver.

SORU:
{question}

KAYNAKLAR:
{context}

CEVAP:
""".strip()

    def _format_appointment_assignments(self, question: str, results: List[Dict]) -> str:
        rows = self._extract_appointment_assignments(results)
        lines = [f"Soru: {question}", ""]

        if not rows:
            lines.extend([
                "Atama kaynakları bulundu; ancak kaynaklarda kişi ve kurum/görev bilgisi okunabilir biçimde yer almıyor.",
                "Kaynak bağlantılarını açarak karar metnini ayrıca kontrol edebilirsiniz.",
                "",
                "Bulunan kaynaklar:",
            ])
            for item_no, item in enumerate(results, start=1):
                metadata = item.get("metadata", {})
                lines.append(f"{item_no}. [{metadata.get('date', '-')}] {metadata.get('title', '-')}")
                lines.append(f"   Kaynak: {metadata.get('item_url', '-')}")
            return "\n".join(lines).strip()

        lines.append("| Tarih | Karar | Kişi | Kurum/Görev | Kaynak |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['date']} | {row['decision']} | {row['person']} | "
                f"{row['position']} | {row['source']} |"
            )
        return "\n".join(lines).strip()

    def _format_appointment_assignments(self, question: str, results: List[Dict]) -> str:
        rows = self._extract_appointment_assignments(results)
        lines = [f"Soru: {question}", ""]

        if not rows:
            unreadable_decisions = [
                item for item in results
                if self._is_appointment_decision_title(item) and self._looks_unreadable_appointment_text(item)
            ]
            if unreadable_decisions:
                lines.extend([
                    "Atama kararı kaynakları bulundu; ancak bu kayıtlardaki PDF metni okunabilir biçimde çıkarılamamış.",
                    "Bu yüzden kişi ve kurum/görev tablosu güvenilir şekilde üretilemiyor.",
                    "İlgili tarihleri yeniden veri çekme ve Qdrant indeksleme işleminden geçirmek gerekir.",
                    "",
                    "Bulunan atama kararı kaynakları:",
                ])
                display_results = unreadable_decisions
            else:
                lines.extend([
                    "Atama kelimesi geçen kaynaklar bulundu; ancak bunlar kişi/kurum atama kararı biçiminde okunabilir metin içermiyor.",
                    "Bu kaynaklar yönetmelik veya genel atanma usulü metni olabilir.",
                    "",
                    "Bulunan kaynaklar:",
                ])
                display_results = results

            for item_no, item in enumerate(display_results, start=1):
                metadata = item.get("metadata", {})
                lines.append(f"{item_no}. [{metadata.get('date', '-')}] {metadata.get('title', '-')}")
                lines.append(f"   Kaynak: {metadata.get('item_url', '-')}")
            return "\n".join(lines).strip()

        lines.append("| Tarih | Karar | Kişi | Kurum/Görev | Kaynak |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['date']} | {row['decision']} | {row['person']} | "
                f"{row['position']} | {row['source']} |"
            )
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
        is_listing = self.is_structured_answer_request(question)
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
- Soru atamalarla ilgiliyse cevabı tablo olarak ver: Tarih | Karar | Kişi | Atandığı kurum/görev | Kaynak.
- Atama tablosunda sadece kaynakta açıkça görünen kişi ve kurum/görev çiftlerini yaz; kaynakta kişi adı veya kurum/görev yoksa "Kaynakta ayrıştırılabilir kişi/kurum bilgisi yok." de.
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

        if self._is_appointment_assignment_request(question):
            rows = self._extract_appointment_assignments(sources)
            if rows:
                return self._format_appointment_assignments(question, sources)
            if LLMClient is not None:
                prompt = self._build_appointment_extraction_prompt(question, sources)
                llm = LLMClient(model=model)
                return llm.generate_answer(prompt)
            return self._format_appointment_assignments(question, sources)

        if LLMClient is None:
            raise RuntimeError("LLMClient yüklenemedi. core/llm_client.py dosyasını ve openai paketini kontrol et.")

        prompt = self.build_prompt_for_llm(question=question, top_k=top_k, sources=sources)
        llm = LLMClient(model=model)
        return llm.generate_answer(prompt)
