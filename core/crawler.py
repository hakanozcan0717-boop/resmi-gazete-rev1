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

    def _build_pdf_item(self, day: dt.date, pdf_url: str, raw_pdf: bytes, fallback_title: str = "") -> Optional[GazetteItem]:
        pdf_dir = ensure_dir(self.data_dir / f"{day:%Y}" / f"{day:%m}")
        fname = safe_filename(Path(urllib.parse.urlparse(pdf_url).path).name or f"{day:%Y%m%d}.pdf")
        pdf_path = pdf_dir / fname
        pdf_path.write_bytes(raw_pdf)
        text = ""
        if PdfReader:
            try:
                reader = PdfReader(str(pdf_path))
                pages = []
                started_at = time.monotonic()
                for page_index, page in enumerate(reader.pages):
                    if page_index >= 30:
                        pages.append("[PDF metin çıkarma ilk 30 sayfa ile sınırlandı.]")
                        break
                    if time.monotonic() - started_at > 30:
                        pages.append("[PDF metin çıkarma süre sınırı nedeniyle durduruldu.]")
                        break
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pass
                text = clean_whitespace("\n".join(pages))
            except Exception as exc:
                print(f"[PDF OKUMA HATA] {pdf_path}: {exc}", file=sys.stderr)
        else:
            text = "PDF indirildi fakat metin çıkarma için pypdf kurulu değil. Kurulum: pip install pypdf"
        title = fallback_title or f"{day:%d.%m.%Y} Resmî Gazete PDF"
        return self._make_item(day, pdf_url, pdf_url, title, text, str(pdf_path))

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
