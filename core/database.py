# -*- coding: utf-8 -*-

import csv
import os
import re
import sqlite3
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras

from config.settings import DEFAULT_DB
from core.models import GazetteItem, SearchResult
from core.utils import clean_whitespace, normalize_tr, now_iso, shorten


class GazetteDB:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.database_url = os.getenv("DATABASE_URL")

        if self.database_url:
            self.backend = "postgres"
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = False
        else:
            self.backend = "sqlite"
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

        self.init_schema()

    def _dict_rows(self, rows):
        if self.backend == "postgres":
            return rows
        return rows

    def init_schema(self) -> None:
        cur = self.conn.cursor()

        if self.backend == "postgres":
            cur.execute("""
            CREATE TABLE IF NOT EXISTS gazette_items (
                id SERIAL PRIMARY KEY,
                date TEXT NOT NULL,
                source_url TEXT NOT NULL,
                item_url TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT,
                institution TEXT,
                content TEXT,
                summary TEXT,
                content_hash TEXT UNIQUE,
                fetched_at TEXT,
                file_path TEXT
            )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_date ON gazette_items(date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_category ON gazette_items(category)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_institution ON gazette_items(institution)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_hash ON gazette_items(content_hash)")

            cur.execute("""
            CREATE TABLE IF NOT EXISTS crawl_log (
                id SERIAL PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                start_date TEXT,
                end_date TEXT,
                inserted_count INTEGER,
                error_count INTEGER,
                notes TEXT
            )
            """)

        else:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS gazette_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                source_url TEXT NOT NULL,
                item_url TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT,
                institution TEXT,
                content TEXT,
                summary TEXT,
                content_hash TEXT UNIQUE,
                fetched_at TEXT,
                file_path TEXT
            )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_date ON gazette_items(date)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_category ON gazette_items(category)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_institution ON gazette_items(institution)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_items_hash ON gazette_items(content_hash)")

            try:
                cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS gazette_fts USING fts5(
                    title, content, summary, category, institution,
                    content='gazette_items', content_rowid='id'
                )
                """)
            except sqlite3.OperationalError:
                pass

            cur.execute("""
            CREATE TABLE IF NOT EXISTS crawl_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                finished_at TEXT,
                start_date TEXT,
                end_date TEXT,
                inserted_count INTEGER,
                error_count INTEGER,
                notes TEXT
            )
            """)

        self.conn.commit()

    def insert_item(self, item: GazetteItem) -> bool:
        cur = self.conn.cursor()

        try:
            if self.backend == "postgres":
                cur.execute("""
                INSERT INTO gazette_items
                (date, source_url, item_url, title, category, institution, content, summary,
                 content_hash, fetched_at, file_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """, (
                    item.date, item.source_url, item.item_url, item.title,
                    item.category, item.institution, item.content, item.summary,
                    item.content_hash, item.fetched_at, item.file_path
                ))

                row = cur.fetchone()
                self.conn.commit()
                return row is not None

            cur.execute("""
            INSERT INTO gazette_items
            (date, source_url, item_url, title, category, institution, content, summary,
             content_hash, fetched_at, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.date, item.source_url, item.item_url, item.title,
                item.category, item.institution, item.content, item.summary,
                item.content_hash, item.fetched_at, item.file_path
            ))

            rowid = cur.lastrowid

            try:
                cur.execute("""
                INSERT INTO gazette_fts(rowid, title, content, summary, category, institution)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (rowid, item.title, item.content, item.summary, item.category, item.institution))
            except sqlite3.OperationalError:
                pass

            self.conn.commit()
            return True

        except Exception:
            self.conn.rollback()
            return False

    def log_crawl(self, start_date: str, end_date: str, inserted: int, errors: int, notes: str = "") -> None:
        cur = self.conn.cursor()

        if self.backend == "postgres":
            cur.execute("""
            INSERT INTO crawl_log(started_at, finished_at, start_date, end_date, inserted_count, error_count, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (now_iso(), now_iso(), start_date, end_date, inserted, errors, notes))
        else:
            cur.execute("""
            INSERT INTO crawl_log(started_at, finished_at, start_date, end_date, inserted_count, error_count, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now_iso(), now_iso(), start_date, end_date, inserted, errors, notes))

        self.conn.commit()

    def _fetchall(self, sql: str, params=()):
        if self.backend == "postgres":
            sql = sql.replace("?", "%s")
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            return cur.fetchall()

        return self.conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params=()):
        if self.backend == "postgres":
            sql = sql.replace("?", "%s")
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            return cur.fetchone()

        return self.conn.execute(sql, params).fetchone()

    def count_items(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS n FROM gazette_items")
        return int(row["n"] if self.backend == "postgres" else row[0])

    def list_items(self, limit: int = 50, offset: int = 0):
        return self._fetchall("""
        SELECT id, date, title, category, institution, summary, item_url
        FROM gazette_items
        ORDER BY date DESC, id DESC
        LIMIT ? OFFSET ?
        """, (limit, offset))

    def get_item(self, item_id: int):
        return self._fetchone("SELECT * FROM gazette_items WHERE id = ?", (item_id,))

    def all_texts(self, limit: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        sql = """
        SELECT id, date, title, category, institution, content, summary, item_url
        FROM gazette_items
        """
        params = []
        filters = []

        if start_date:
            filters.append("date >= ?")
            params.append(start_date)
        if end_date:
            filters.append("date <= ?")
            params.append(end_date)

        if filters:
            sql += " WHERE " + " AND ".join(filters)

        sql += " ORDER BY date DESC"

        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))

        return self._fetchall(sql, tuple(params))

    def search(self, query: str, limit: int = 30) -> List[SearchResult]:
        query = clean_whitespace(query)

        if not query:
            return []

        results = []
        terms = [normalize_tr(t) for t in query.split() if len(t) > 1]

        rows = self._fetchall("""
        SELECT id, date, title, category, institution, source_url, item_url, content, summary
        FROM gazette_items
        ORDER BY date DESC
        LIMIT 5000
        """)

        for r in rows:
            title = r["title"] or ""
            content = r["content"] or ""
            hay = normalize_tr(title + " " + content)
            score = sum(hay.count(t) for t in terms)

            if score > 0:
                snippet = self._make_snippet(content or r["summary"] or "", terms)
                results.append(SearchResult(
                    r["id"],
                    r["date"],
                    r["title"],
                    r["category"] or "",
                    r["institution"] or "",
                    r["source_url"],
                    r["item_url"],
                    float(score),
                    snippet
                ))

        results.sort(key=lambda x: (x.score, x.date), reverse=True)
        return results[:limit]

    def _make_snippet(self, content: str, terms: List[str], radius: int = 180) -> str:
        norm = normalize_tr(content)
        positions = [norm.find(t) for t in terms if norm.find(t) >= 0]

        if not positions:
            return shorten(content, 360)

        pos = min(positions)
        return clean_whitespace(content[max(0, pos-radius):min(len(content), pos+radius)])

    def stats_by_category(self) -> List[Tuple[str, int]]:
        rows = self._fetchall("""
        SELECT COALESCE(category, 'Diğer') AS category, COUNT(*) AS n
        FROM gazette_items
        GROUP BY COALESCE(category, 'Diğer')
        ORDER BY n DESC
        """)

        return [(r["category"], r["n"]) for r in rows]

    def stats_by_date(self) -> List[Tuple[str, int]]:
        rows = self._fetchall("""
        SELECT date, COUNT(*) AS n
        FROM gazette_items
        GROUP BY date
        ORDER BY date DESC
        """)

        return [(r["date"], r["n"]) for r in rows]

    def list_dates(self) -> List[Tuple[str, int]]:
        return self.stats_by_date()

    def stats_by_institution(self, limit: int = 20) -> List[Tuple[str, int]]:
        rows = self._fetchall("""
        SELECT institution, COUNT(*) AS n
        FROM gazette_items
        WHERE institution IS NOT NULL AND institution != ''
        GROUP BY institution
        ORDER BY n DESC
        LIMIT ?
        """, (limit,))

        return [(r["institution"], r["n"]) for r in rows]

    def export_csv(self, out_path: str) -> None:
        rows = self._fetchall("""
        SELECT id, date, title, category, institution, summary, source_url, item_url, fetched_at
        FROM gazette_items
        ORDER BY date DESC, id DESC
        """)

        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "date", "title", "category", "institution",
                "summary", "source_url", "item_url", "fetched_at"
            ])

            for r in rows:
                writer.writerow([
                    r["id"], r["date"], r["title"], r["category"],
                    r["institution"], r["summary"], r["source_url"],
                    r["item_url"], r["fetched_at"]
                ])
