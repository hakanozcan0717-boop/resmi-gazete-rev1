# -*- coding: utf-8 -*-
import argparse
import datetime as dt
import hashlib
import html
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List
from core.constants import TURKISH_STOPWORDS, CATEGORY_KEYWORDS

def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()

def parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("Tarih formatı YYYY-MM-DD olmalıdır. Örnek: 2026-05-11")

def date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def safe_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^\w\-.ğüşöçıİĞÜŞÖÇ ]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] or "dosya"

def clean_whitespace(text: str) -> str:
    text = html.unescape(text or "").replace("\u00a0", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def strip_html_fallback(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    raw_html = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw_html)
    raw_html = re.sub(r"(?i)<br\s*/?>", "\n", raw_html)
    raw_html = re.sub(r"(?i)</p>", "\n", raw_html)
    raw_html = re.sub(r"(?i)</div>", "\n", raw_html)
    raw_html = re.sub(r"<[^>]+>", " ", raw_html)
    return clean_whitespace(raw_html)

def normalize_tr(text: str) -> str:
    table = str.maketrans("IİŞĞÜÖÇıişğüöç", "iişgüöçıişğüöç")
    return text.translate(table).lower()

def tokenize_tr(text: str) -> List[str]:
    text = normalize_tr(text)
    tokens = re.findall(r"[a-zçğıöşüâîû0-9]{2,}", text, flags=re.UNICODE)
    return [t for t in tokens if t not in TURKISH_STOPWORDS and not t.isdigit()]

def split_sentences(text: str) -> List[str]:
    text = clean_whitespace(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]

def shorten(text: str, length: int = 240) -> str:
    text = clean_whitespace(text)
    return text if len(text) <= length else text[: length - 3].rstrip() + "..."

def guess_category(title: str, text: str) -> str:
    haystack = normalize_tr((title or "") + " " + (text or "")[:5000])
    scores = {}
    for cat, words in CATEGORY_KEYWORDS.items():
        score = sum(haystack.count(normalize_tr(w)) for w in words)
        if score:
            scores[cat] = score
    return max(scores.items(), key=lambda x: x[1])[0] if scores else "Diğer"

def extract_institution(title: str, text: str) -> str:
    sample = (title or "") + "\n" + (text or "")[:3000]
    patterns = [
        r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+BAKANLIĞI)",
        r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+BAŞKANLIĞI)",
        r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+KURUMU)",
        r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+ÜNİVERSİTESİ)",
        r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+MÜDÜRLÜĞÜ)",
    ]
    for pat in patterns:
        m = re.search(pat, sample)
        if m:
            return clean_whitespace(m.group(1)).title()
    return ""

def extractive_summary(text: str, max_sentences: int = 4) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    freq = Counter(tokenize_tr(text))
    if not freq:
        return " ".join(sentences[:max_sentences])
    scored = []
    for idx, sent in enumerate(sentences[:120]):
        tokens = tokenize_tr(sent)
        if not tokens:
            continue
        score = sum(freq[t] for t in tokens) / math.sqrt(len(tokens))
        if idx < 10:
            score *= 1.15
        scored.append((score, idx, sent))
    best = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(s for _, _, s in sorted(best, key=lambda x: x[1]))
def clean_extracted_text(text: str) -> str:
    """
    PDF/HTML metin çıkarma sonrası oluşan bozuk boşlukları ve satırları düzeltir.

    Örnek düzelttiği şeyler:
    - "Ortaklığ ı" -> "Ortaklığı"
    - "no .lu" -> "no.lu"
    - "m 2" -> "m2"
    - gereksiz satır kırılımları
    """

    text = text or ""

    # Bozuk kodlama ihtimalleri
    replacements = {
        "Ý": "İ",
        "ý": "ı",
        "Þ": "Ş",
        "þ": "ş",
        "Ð": "Ğ",
        "ð": "ğ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Satır sonlarını normalize et
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Tireyle bölünmüş kelimeleri birleştir
    # Örn: "yönet-\nmelik" -> "yönetmelik"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text, flags=re.UNICODE)

    # Satır sonu yüzünden ikiye bölünmüş kelimeleri birleştir
    # Örn: "ilç e" gibi PDF kaynaklı bazı hataları tamamen çözemeyebilir ama azaltır.
    text = re.sub(r"([a-zçğıöşü])\s+([ıiuüaeo])\b", r"\1\2", text, flags=re.IGNORECASE)

    # Noktalama öncesi boşlukları kaldır
    # Örn: "no .lu" -> "no.lu"
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # m 2, km 2 gibi ifadeleri birleştir
    text = re.sub(r"\b(m|km|cm|mm)\s+2\b", r"\1²", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(m|km|cm|mm)\s+3\b", r"\1³", text, flags=re.IGNORECASE)

    # Çok fazla boşluğu azalt
    text = re.sub(r"[ \t]+", " ", text)

    # Çok fazla boş satırı azalt
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()