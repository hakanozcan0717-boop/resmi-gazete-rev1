# -*- coding: utf-8 -*-
import sys
import time
from typing import Tuple
from config.settings import USER_AGENT
try:
    import requests
except Exception:
    requests = None

class HttpClient:
    def __init__(self, timeout: int = 60, sleep: float = 1.5):
        self.timeout = timeout
        self.sleep = sleep
        self.last_request_at = 0.0
        self.session = None
        if requests:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT})

    def get(self, url: str) -> Tuple[int, bytes, str]:
        elapsed = time.time() - self.last_request_at
        if elapsed < self.sleep:
            time.sleep(self.sleep - elapsed)
        self.last_request_at = time.time()
        urls_to_try = [url]
        if "https://www.resmigazete.gov.tr" in url:
            urls_to_try.append(url.replace("https://www.resmigazete.gov.tr", "https://resmigazete.gov.tr"))
        elif "https://resmigazete.gov.tr" in url:
            urls_to_try.append(url.replace("https://resmigazete.gov.tr", "https://www.resmigazete.gov.tr"))
        urls_to_try = list(dict.fromkeys(urls_to_try))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
        }
        if requests:
            for try_url in urls_to_try:
                for attempt in range(1, 4):
                    try:
                        print(f"[DENEME] {try_url} - {attempt}. deneme")
                        r = self.session.get(try_url, timeout=(self.timeout, self.timeout), allow_redirects=True, headers=headers)
                        content_type = r.headers.get("content-type", "")
                        if r.status_code == 200 and r.content:
                            return r.status_code, r.content, content_type
                        print(f"[UYARI] HTTP durum kodu: {r.status_code} - {try_url}", file=sys.stderr)
                    except Exception as exc:
                        print(f"[HTTP HATA] {try_url}: {exc}", file=sys.stderr)
                        time.sleep(2 * attempt)
        for try_url in urls_to_try:
            try:
                print(f"[URLLIB DENEME] {try_url}")
                import urllib.request
                req = urllib.request.Request(try_url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.status, resp.read(), resp.headers.get("content-type", "")
            except Exception as exc:
                print(f"[URLLIB HATA] {try_url}: {exc}", file=sys.stderr)
        return 0, b"", ""
