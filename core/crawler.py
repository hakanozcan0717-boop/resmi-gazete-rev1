# -*- coding: utf-8 -*-
import datetime as dt
import html
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple
from config.settings import BASE_URL, DEFAULT_DATA_DIR
from core.http_client import HttpClient
from core.models import GazetteItem
from core.utils import clean_whitespace, ensure_dir, extract_institution, extractive_summary, guess_category, now_iso, safe_filename, sha256_text, strip_html_fallback, clean_extracted_text

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    import fitz
except Exception:
    fitz = None

class OfficialGazetteCrawler:
    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        timeout: int = 30,
        sleep: float = 0.5,
        retries: int = 2,
        max_request_seconds: int = 120,
    ):
        self.data_dir = ensure_dir(data_dir)
        self.client = HttpClient(
            timeout=timeout,
            sleep=sleep,
            retries=retries,
            max_request_seconds=max_request_seconds,
        )

    @staticmethod
    def daily_html_url(day: dt.date) -> str:
        return f"{BASE_URL}/eskiler/{day:%Y}/{day:%m}/{day:%Y%m%d}.htm"

    @staticmethod
    def daily_pdf_url(day: dt.date) -> str:
        return f"{BASE_URL}/eskiler/{day:%Y}/{day:%m}/{day:%Y%m%d}.pdf"

    def fetch_day(self, day: dt.date) -> List[GazetteItem]:
        html_url = self.daily_html_url(day)
        pdf_url = self.daily_pdf_url(day)
        print(f"[TARA] {day} -> {html_url}")
        status, raw, _ = self.client.get(html_url)
        items: List[GazetteItem] = []
        if status == 200 and raw:
            try:
                items.extend(self._parse_daily_html(day, html_url, raw))
            except Exception as exc:
                print(f"[PARSE HATA] {html_url}: {exc}", file=sys.stderr)
        if not items:
            print(f"[BİLGİ] HTML içerik bulunamadı veya boş. PDF deneniyor: {pdf_url}")
            pstatus, praw, _ = self.client.get(pdf_url)
            if pstatus == 200 and praw:
                item = self._build_pdf_item(day, pdf_url, praw)
                if item:
                    items.append(item)
        print(f"[SONUÇ] {day}: {len(items)} kayıt")
        return items

    def _parse_daily_html(self, day: dt.date, html_url: str, raw: bytes) -> List[GazetteItem]:
        decoded = self._decode_bytes(raw)
        links = self._extract_links(decoded, html_url)
        day_key = f"{day:%Y%m%d}"
        candidate_links = []
        for title, href in links:
            if day_key in href and (href.endswith(".htm") or href.endswith(".html") or href.endswith(".pdf")):
                if href == self.daily_pdf_url(day):
                    continue
                candidate_links.append((title, href))
        if not candidate_links:
            text = self._html_to_text(decoded)
            title = self._extract_title_from_html(decoded) or f"{day:%d.%m.%Y} Resmî Gazete"
            return [self._make_item(day, html_url, html_url, title, text, "")] if len(text) > 200 else []
        seen = set()
        unique_links = []
        for title, href in candidate_links:
            if href not in seen:
                unique_links.append((title, href))
                seen.add(href)
        items = []
        for idx, (link_title, item_url) in enumerate(unique_links, start=1):
            status, raw_item, ctype = self.client.get(item_url)
            if status != 200 or not raw_item:
                continue
            if item_url.lower().endswith(".pdf") or "pdf" in ctype.lower():
                item = self._build_pdf_item(day, item_url, raw_item, fallback_title=link_title)
                if item and self._needs_html_fallback(item.content):
                    html_item = self._build_companion_html_item(day, html_url, item_url, link_title)
                    if html_item:
                        item = html_item
            else:
                item_html = self._decode_bytes(raw_item)
                text = self._html_to_text(item_html)
                title = self._extract_title_from_html(item_html) or link_title or f"Belge {idx}"
                item = self._make_item(day, html_url, item_url, title, text, "")
            if item and len(item.content) > 80:
                items.append(item)
        return items

    def _extract_links(self, raw_html: str, base_url: str) -> List[Tuple[str, str]]:
        links = []
        if BeautifulSoup:
            soup = BeautifulSoup(raw_html, "html.parser")
            for a in soup.find_all("a", href=True):
                title = clean_whitespace(a.get_text(" ", strip=True))
                href = urllib.parse.urljoin(base_url, a["href"])
                links.append((title, href))
        else:
            pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
            for m in re.finditer(pattern, raw_html, re.I | re.S):
                href = urllib.parse.urljoin(base_url, html.unescape(m.group(1)))
                title = strip_html_fallback(m.group(2))
                links.append((title, href))
        return links

    def _html_to_text(self, raw_html: str) -> str:
        if BeautifulSoup:
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return clean_whitespace(soup.get_text("\n", strip=True))
        return strip_html_fallback(raw_html)

    def _extract_title_from_html(self, raw_html: str) -> str:
        if BeautifulSoup:
            soup = BeautifulSoup(raw_html, "html.parser")
            for selector in ["h1", "h2", "h3", "title"]:
                tag = soup.find(selector)
                if tag:
                    text = clean_whitespace(tag.get_text(" ", strip=True))
                    if text:
                        return text[:300]
        m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
        return strip_html_fallback(m.group(1))[:300] if m else ""

    def _build_companion_html_item(
        self,
        day: dt.date,
        source_url: str,
        pdf_url: str,
        fallback_title: str = "",
    ) -> Optional[GazetteItem]:
        for html_url in self._companion_html_urls(pdf_url):
            status, raw_html, ctype = self.client.get(html_url)
            if status != 200 or not raw_html or "pdf" in ctype.lower():
                continue

            decoded = self._decode_bytes(raw_html)
            text = clean_extracted_text(self._html_to_text(decoded))
            if len(text) < 120:
                continue

            title = self._extract_title_from_html(decoded) or fallback_title or f"Belge {day:%Y-%m-%d}"
            print(f"[HTML FALLBACK] PDF yerine HTML kullanıldı: {html_url}")
            return self._make_item(day, source_url, html_url, title, text, "")

        return None

    def _companion_html_urls(self, pdf_url: str) -> List[str]:
        parsed = urllib.parse.urlparse(pdf_url)
        path = parsed.path or ""
        if not path.lower().endswith(".pdf"):
            return []

        base_path = path[:-4]
        candidates = []
        for suffix in [".htm", ".html"]:
            candidate = urllib.parse.urlunparse(parsed._replace(path=base_path + suffix))
            candidates.append(candidate)

        return candidates

    def _build_pdf_item(self, day: dt.date, pdf_url: str, raw_pdf: bytes, fallback_title: str = "") -> Optional[GazetteItem]:
        pdf_dir = ensure_dir(self.data_dir / f"{day:%Y}" / f"{day:%m}")
        fname = safe_filename(Path(urllib.parse.urlparse(pdf_url).path).name or f"{day:%Y%m%d}.pdf")
        pdf_path = pdf_dir / fname
        pdf_path.write_bytes(raw_pdf)
        candidates = []

        pymupdf_text = self._extract_pdf_text_pymupdf(pdf_path)
        if pymupdf_text:
            candidates.append(("pymupdf", pymupdf_text))

        if not self._is_good_pdf_text(pymupdf_text):
            pypdf_text = self._extract_pdf_text_pypdf(pdf_path)
            if pypdf_text:
                candidates.append(("pypdf", pypdf_text))

        if candidates:
            extractor_name, text = max(candidates, key=lambda item: self._pdf_text_quality_score(item[1]))
            print(f"[PDF METIN] {pdf_path.name}: {extractor_name} kullanıldı")
        else:
            text = "PDF indirildi fakat metin çıkarılamadı. PyMuPDF veya pypdf kurulumunu kontrol edin."
        title = fallback_title or f"{day:%d.%m.%Y} Resmî Gazete PDF"
        return self._make_item(day, pdf_url, pdf_url, title, text, str(pdf_path))

    def _extract_pdf_text_pymupdf(self, pdf_path: Path, max_pages: int = 30, max_seconds: int = 30) -> str:
        if not fitz:
            return ""

        try:
            pages = []
            started_at = time.monotonic()
            with fitz.open(str(pdf_path)) as doc:
                for page_index, page in enumerate(doc):
                    if page_index >= max_pages:
                        pages.append("[PDF metin çıkarma ilk 30 sayfa ile sınırlandı.]")
                        break
                    if time.monotonic() - started_at > max_seconds:
                        pages.append("[PDF metin çıkarma süre sınırı nedeniyle durduruldu.]")
                        break
                    try:
                        pages.append(page.get_text("text") or "")
                    except Exception:
                        pass
            return clean_whitespace("\n".join(pages))
        except Exception as exc:
            print(f"[PDF PYMUPDF HATA] {pdf_path}: {exc}", file=sys.stderr)
            return ""

    def _extract_pdf_text_pypdf(self, pdf_path: Path, max_pages: int = 30, max_seconds: int = 30) -> str:
        if not PdfReader:
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            started_at = time.monotonic()
            for page_index, page in enumerate(reader.pages):
                if page_index >= max_pages:
                    pages.append("[PDF metin çıkarma ilk 30 sayfa ile sınırlandı.]")
                    break
                if time.monotonic() - started_at > max_seconds:
                    pages.append("[PDF metin çıkarma süre sınırı nedeniyle durduruldu.]")
                    break
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pass
            return clean_whitespace("\n".join(pages))
        except Exception as exc:
            print(f"[PDF PYPDF HATA] {pdf_path}: {exc}", file=sys.stderr)
            return ""

    def _pdf_text_quality_score(self, text: str) -> float:
        cleaned = clean_extracted_text(text or "")
        if not cleaned:
            return 0.0

        length = len(cleaned)
        alpha = sum(1 for char in cleaned if char.isalpha())
        control = sum(1 for char in cleaned if ord(char) < 32 and char not in "\n\r\t")
        noisy = sum(1 for char in cleaned if char in "#$%&*<=>?@[\\]^_`{|}~\ufffd\u25a1")
        turkish_hits = sum(
            cleaned.lower().count(word)
            for word in ["resmî", "gazete", "madde", "karar", "yönetmelik", "tebliğ", "atama"]
        )

        return (
            min(length, 50000) / 100
            + alpha * 0.5
            + turkish_hits * 200
            - noisy * 5
            - control * 50
        )

    def _is_good_pdf_text(self, text: str) -> bool:
        cleaned = clean_extracted_text(text or "")
        if len(cleaned) < 300:
            return False

        alpha = sum(1 for char in cleaned if char.isalpha())
        noisy = sum(1 for char in cleaned if char in "#$%&*<=>?@[\\]^_`{|}~\ufffd\u25a1")
        if alpha < 180:
            return False

        return noisy / max(len(cleaned), 1) < 0.08 and self._pdf_text_quality_score(cleaned) >= 800

    def _needs_html_fallback(self, text: str) -> bool:
        cleaned = clean_extracted_text(text or "")
        if len(cleaned) < 120:
            return True

        alpha = sum(1 for char in cleaned if char.isalpha())
        noisy = sum(1 for char in cleaned if char in "#$%&*<=>?@[\\]^_`{|}~\ufffd\u25a1")
        return alpha < 80 or noisy / max(len(cleaned), 1) >= 0.12

    def _make_item(self, day: dt.date, source_url: str, item_url: str, title: str, content: str, file_path: str) -> GazetteItem:
        content = clean_extracted_text(clean_whitespace(content))
        title = clean_whitespace(title) or "Başlıksız Belge"
        category = guess_category(title, content)
        institution = extract_institution(title, content)
        summary = extractive_summary(content, max_sentences=4)
        return GazetteItem(
            date=day.isoformat(), source_url=source_url, item_url=item_url,
            title=title[:500], category=category, institution=institution[:250],
            content=content, summary=summary,
            content_hash=sha256_text(item_url + "\n" + content[:10000]),
            fetched_at=now_iso(), file_path=file_path,
        )

    def _decode_bytes(self, raw: bytes) -> str:
        for enc in ["utf-8", "windows-1254", "iso-8859-9", "latin-1"]:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")
