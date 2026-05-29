#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resmî Gazete Analiz ve Takip Sistemi - Ana Çalıştırma Dosyası

RAG ve LLM komutları eklenmiş sürüm.
"""

import argparse
import textwrap
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

from config.settings import DEFAULT_DATA_DIR, DEFAULT_DB, DEFAULT_HOST, DEFAULT_PORT
from core.commands import (
    cmd_analyze,
    cmd_crawl,
    cmd_export,
    cmd_import_sqlite,
    cmd_rag_ask,
    cmd_rag_index,
    cmd_rag_llm,
    cmd_search,
    cmd_serve,
    cmd_show,
)
from core.utils import parse_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Resmî Gazete veri çekme, analiz, arama, RAG ve LLM sistemi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Örnekler:
          python main.py crawl --days 7
          python main.py search --query "ihale yönetmeliği"
          python main.py analyze --json-out exports/analiz.json
          python main.py serve

        RAG örnekleri:
          python main.py rag-index
          python main.py rag-ask --question "İhale ile ilgili düzenlemeler nelerdir?"
          python main.py rag-llm --question "Vergiyle ilgili kararları özetle"
        """),
    )

    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite veritabanı yolu. Varsayılan: {DEFAULT_DB}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_crawl = sub.add_parser("crawl", help="Resmî Gazete tarih aralığı indirir.")
    p_crawl.add_argument("--start", help="Başlangıç tarihi YYYY-MM-DD")
    p_crawl.add_argument("--end", help="Bitiş tarihi YYYY-MM-DD")
    p_crawl.add_argument("--days", type=int, help="Bugünden geriye kaç gün indirilsin?")
    p_crawl.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="PDF/ham veri klasörü")
    p_crawl.add_argument("--timeout", type=int, default=60)
    p_crawl.add_argument("--retries", type=int, default=2, help="Her URL için deneme sayısı")
    p_crawl.add_argument("--max-request-seconds", type=int, default=120, help="Tek URL için toplam süre sınırı")
    p_crawl.add_argument("--sleep", type=float, default=0.6, help="İstekler arası bekleme saniyesi")
    p_crawl.add_argument("--debug", action="store_true")
    p_crawl.add_argument("--fail-on-empty", action="store_true", help="Hiç belge bulunamazsa komutu hata ile bitir")
    p_crawl.add_argument("--fail-on-errors", action="store_true", help="Gün bazlı hata varsa komutu hata ile bitir")
    p_crawl.set_defaults(func=cmd_crawl)

    p_search = sub.add_parser("search", help="Veritabanında arama yapar.")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=30)
    p_search.set_defaults(func=cmd_search)

    p_analyze = sub.add_parser("analyze", help="Genel analiz raporu üretir.")
    p_analyze.add_argument("--json-out", help="Raporu JSON dosyasına kaydet")
    p_analyze.set_defaults(func=cmd_analyze)

    p_export = sub.add_parser("export", help="CSV raporu üretir.")
    p_export.add_argument("--out", default="exports/resmi_gazete_rapor.csv")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import-sqlite", help="SQLite verilerini hedef veritabanına aktarır.")
    p_import.add_argument("--sqlite-db", default=DEFAULT_DB, help="Kaynak SQLite veritabanı yolu")
    p_import.add_argument("--start", help="Başlangıç tarihi YYYY-MM-DD")
    p_import.add_argument("--end", help="Bitiş tarihi YYYY-MM-DD")
    p_import.add_argument("--limit", type=int, default=None, help="En fazla kaç kayıt aktarılsın?")
    p_import.set_defaults(func=cmd_import_sqlite)

    p_show = sub.add_parser("show", help="Tek bir kaydı detaylı gösterir.")
    p_show.add_argument("--id", type=int, required=True)
    p_show.add_argument("--full", action="store_true", help="Tam metni de göster")
    p_show.set_defaults(func=cmd_show)

    p_serve = sub.add_parser("serve", help="Web paneli başlatır.")
    p_serve.add_argument("--host", default=DEFAULT_HOST)
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_rag_index = sub.add_parser("rag-index", help="SQLite belgelerini RAG vektör veritabanına işler.")
    p_rag_index.add_argument("--vector-db", default="vector_db", help="Vektör veritabanı klasörü")
    p_rag_index.add_argument("--limit", type=int, default=None, help="Kaç belge işlensin? Boşsa tüm belgeler")
    p_rag_index.add_argument("--start", help="Başlangıç tarihi YYYY-MM-DD")
    p_rag_index.add_argument("--end", help="Bitiş tarihi YYYY-MM-DD")
    p_rag_index.add_argument("--days", type=int, help="Bugünden geriye kaç gün indekslensin?")
    p_rag_index.add_argument("--chunk-size", type=int, default=1000, help="Her metin parçasının yaklaşık karakter sayısı")
    p_rag_index.add_argument("--overlap", type=int, default=150, help="Parçalar arası ortak karakter sayısı")
    p_rag_index.set_defaults(func=cmd_rag_index)

    p_rag_ask = sub.add_parser("rag-ask", help="LLM kullanmadan RAG ile ilgili kaynak parçalarını getirir.")
    p_rag_ask.add_argument("--vector-db", default="vector_db")
    p_rag_ask.add_argument("--question", required=True)
    p_rag_ask.add_argument("--top-k", type=int, default=5)
    p_rag_ask.set_defaults(func=cmd_rag_ask)

    p_rag_llm = sub.add_parser("rag-llm", help="RAG kaynaklarını bulur ve LLM ile cevap üretir.")
    p_rag_llm.add_argument("--vector-db", default="vector_db")
    p_rag_llm.add_argument("--question", required=True)
    p_rag_llm.add_argument("--top-k", type=int, default=5)
    p_rag_llm.add_argument("--model", default=None, help="OpenAI model adı. Boşsa .env içindeki OPENAI_MODEL kullanılır.")
    p_rag_llm.set_defaults(func=cmd_rag_llm)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "crawl":
        if not args.days and (not args.start or not args.end):
            raise SystemExit("crawl için ya --days ya da --start ve --end birlikte verilmelidir.")
        if args.days is not None and args.days <= 0:
            raise SystemExit("--days pozitif olmalıdır.")
        if args.retries <= 0:
            raise SystemExit("--retries pozitif olmalıdır.")
        if args.max_request_seconds <= 0:
            raise SystemExit("--max-request-seconds pozitif olmalıdır.")
        if args.start and args.end:
            start = parse_date(args.start)
            end = parse_date(args.end)
            if start > end:
                raise SystemExit("Başlangıç tarihi bitiş tarihinden büyük olamaz.")
    elif args.command == "rag-index":
        if args.days is not None and args.days <= 0:
            raise SystemExit("--days pozitif olmalıdır.")
        if args.days and (args.start or args.end):
            raise SystemExit("rag-index için --days ile --start/--end birlikte kullanılmamalıdır.")
        if bool(args.start) != bool(args.end):
            raise SystemExit("rag-index için --start ve --end birlikte verilmelidir.")
        if args.start and args.end:
            start = parse_date(args.start)
            end = parse_date(args.end)
            if start > end:
                raise SystemExit("Başlangıç tarihi bitiş tarihinden büyük olamaz.")
    elif args.command == "import-sqlite":
        if args.limit is not None and args.limit <= 0:
            raise SystemExit("--limit pozitif olmalıdır.")
        if bool(args.start) != bool(args.end):
            raise SystemExit("import-sqlite için --start ve --end birlikte verilmelidir.")
        if args.start and args.end:
            start = parse_date(args.start)
            end = parse_date(args.end)
            if start > end:
                raise SystemExit("Başlangıç tarihi bitiş tarihinden büyük olamaz.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
