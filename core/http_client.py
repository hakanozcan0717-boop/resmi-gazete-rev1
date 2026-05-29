# -*- coding: utf-8 -*-
import sys
import time
from typing import Dict, Optional, Tuple
from config.settings import USER_AGENT
try:
    import requests
except Exception:
    requests = None

class HttpClient:
    def __init__(self, timeout: int = 60, sleep: float = 1.5, retries: int = 2, max_request_seconds: Optional[int] = None):
        self.timeout = timeout
        self.connect_timeout = min(max(timeout, 5), 30)
        self.pdf_read_timeout = min(max(timeout, 90), 180)
        self.retries = max(1, retries)
        self.max_request_seconds = max_request_seconds or max(timeout * 2, 120)
        self.sleep = sleep
        self.last_request_at = 0.0
        self.session = None
        if requests:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT})

    def get(self, url: str) -> Tuple[int, bytes, str]:
        self._wait_before_request()
        started_at = time.monotonic()
        urls_to_try = self._alternate_hosts(url)
        headers = self._headers()
        if requests:
            for try_url in urls_to_try:
                if self._request_deadline_exceeded(started_at):
                    print(f"[HTTP ATLA] Toplam istek süresi aşıldı: {url}", file=sys.stderr)
                    return 0, b"", ""
                for attempt in range(1, self.retries + 1):
                    if self._request_deadline_exceeded(started_at):
                        print(f"[HTTP ATLA] Toplam istek süresi aşıldı: {url}", file=sys.stderr)
                        return 0, b"", ""
                    try:
                        print(f"[DENEME] {try_url} - {attempt}. deneme")
                        status, content, content_type = self._requests_get(try_url, headers)
                        if status == 200 and content:
                            return status, content, content_type
                        print(f"[UYARI] HTTP durum kodu: {status} - {try_url}", file=sys.stderr)
                        if status not in {408, 429, 500, 502, 503, 504}:
                            break
                    except Exception as exc:
                        print(f"[HTTP HATA] {try_url}: {exc}", file=sys.stderr)
                    time.sleep(min(2 * attempt, 5))
        for try_url in urls_to_try:
            if self._request_deadline_exceeded(started_at):
                print(f"[URLLIB ATLA] Toplam istek süresi aşıldı: {url}", file=sys.stderr)
                return 0, b"", ""
            try:
                print(f"[URLLIB DENEME] {try_url}")
                import urllib.request
                req = urllib.request.Request(try_url, headers=headers)
                with urllib.request.urlopen(req, timeout=self._read_timeout_for(try_url)) as resp:
                    return resp.status, resp.read(), resp.headers.get("content-type", "")
            except Exception as exc:
                print(f"[URLLIB HATA] {try_url}: {exc}", file=sys.stderr)
        return 0, b"", ""

    def _request_deadline_exceeded(self, started_at: float) -> bool:
        return (time.monotonic() - started_at) >= self.max_request_seconds

    def _wait_before_request(self) -> None:
        elapsed = time.time() - self.last_request_at
        if elapsed < self.sleep:
            time.sleep(self.sleep - elapsed)
        self.last_request_at = time.time()

    def _alternate_hosts(self, url: str):
        urls_to_try = [url]
        if "https://www.resmigazete.gov.tr" in url:
            urls_to_try.append(url.replace("https://www.resmigazete.gov.tr", "https://resmigazete.gov.tr"))
        elif "https://resmigazete.gov.tr" in url:
            urls_to_try.append(url.replace("https://resmigazete.gov.tr", "https://www.resmigazete.gov.tr"))
        return list(dict.fromkeys(urls_to_try))

    def _headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
        }

    def _requests_get(self, url: str, headers: Dict[str, str]) -> Tuple[int, bytes, str]:
        read_timeout = self._read_timeout_for(url)
        timeout = (self.connect_timeout, read_timeout)
        with self.session.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True) as response:
            content_type = response.headers.get("content-type", "")
            if response.status_code != 200:
                response.close()
                return response.status_code, b"", content_type

            chunks = []
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    chunks.append(chunk)
            return response.status_code, b"".join(chunks), content_type

    def _read_timeout_for(self, url: str) -> int:
        if url.lower().endswith(".pdf"):
            return self.pdf_read_timeout
        return self.timeout
