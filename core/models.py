# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class GazetteItem:
    date: str
    source_url: str
    item_url: str
    title: str
    category: str
    institution: str
    content: str
    summary: str
    content_hash: str
    fetched_at: str
    file_path: str = ""

@dataclass
class SearchResult:
    id: int
    date: str
    title: str
    category: str
    institution: str
    source_url: str
    item_url: str
    score: float
    snippet: str
