# -*- coding: utf-8 -*-
"""
Komut satırı işlemleri.

Bu sürümde RAG ve LLM komutları da eklenmiştir:
- rag-index
- rag-ask
- rag-llm
"""

import datetime as dt
import json
import sqlite3
import sys
import traceback
from pathlib import Path

from config.settings import APP_NAME
from core.analyzer import GazetteAnalyzer
from core.crawler import OfficialGazetteCrawler
from core.database import GazetteDB
from core.models import GazetteItem
from core.rag import RAGEngine
from core.utils import date_range, parse_date
from web.app import create_app


def _row_to_gazette_item(row) -> GazetteItem:
    return GazetteItem(
        date=row["date"] or "",
        source_url=row["source_url"] or "",
        item_url=row["item_url"] or "",
        title=row["title"] or "",
        category=row["category"] or "",
        institution=row["institution"] or "",
        content=row["content"] or "",
        summary=row["summary"] or "",
        content_hash=row["content_hash"] or "",
        fetched_at=row["fetched_at"] or "",
        file_path=row["file_path"] or "",
    )


def cmd_crawl(args) -> None:
    db = GazetteDB(args.db)
    crawler = OfficialGazetteCrawler(
        args.data_dir,
        timeout=args.timeout,
        sleep=args.sleep,
        retries=args.retries,
        max_request_seconds=args.max_request_seconds,
    )

    if args.days:
        end = dt.date.today()
        start = end - dt.timedelta(days=args.days - 1)
    else:
        start = parse_date(args.start)
        end = parse_date(args.end)

    inserted = 0
    errors = 0

    for day in date_range(start, end):
        try:
            items = crawler.fetch_day(day)
            for item in items:
                if db.insert_item(item):
                    inserted += 1
        except KeyboardInterrupt:
            print("\n[İPTAL] Kullanıcı tarafından durduruldu.")
            break
        except Exception as exc:
            errors += 1
            print(f"[GÜN HATA] {day}: {exc}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()

    db.log_crawl(start.isoformat(), end.isoformat(), inserted, errors)
    print(f"\nTamamlandı. Yeni eklenen kayıt: {inserted}, hata: {errors}, toplam kayıt: {db.count_items()}")


def cmd_search(args) -> None:
    db = GazetteDB(args.db)
    results = db.search(args.query, limit=args.limit)

    if not results:
        print("Sonuç bulunamadı.")
        return

    for i, r in enumerate(results, start=1):
        print("=" * 90)
        print(f"{i}. [{r.date}] {r.title}")
        print(f"Kategori: {r.category} | Kurum: {r.institution or '-'} | Skor: {r.score:.2f}")
        print(f"URL: {r.item_url}")
        print(f"Özet/Snippet: {r.snippet}")


def cmd_analyze(args) -> None:
    db = GazetteDB(args.db)
    analyzer = GazetteAnalyzer(db)
    report = analyzer.build_report()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON rapor yazıldı: {args.json_out}")

    print("\n" + APP_NAME)
    print("=" * len(APP_NAME))
    print(f"Toplam belge: {report['total_documents']}")

    print("\nKategori dağılımı:")
    for cat, n in report["category_distribution"]:
        print(f"  - {cat}: {n}")

    print("\nEn sık kurumlar:")
    for inst, n in report["institution_distribution"]:
        print(f"  - {inst}: {n}")

    print("\nEn sık kelimeler:")
    for word, n in report["top_words"][:30]:
        print(f"  - {word}: {n}")

    print("\nMetin uzunluğu:")
    for k, v in report["content_length"].items():
        print(f"  - {k}: {v}")

    print("\nKonu kümeleri:")
    if isinstance(report["topic_clusters"], list):
        for c in report["topic_clusters"]:
            print(f"  Küme {c['cluster']} ({c['size']} belge): {', '.join(c['keywords'])}")
    else:
        print(" ", report["topic_clusters"])


def cmd_export(args) -> None:
    db = GazetteDB(args.db)
    db.export_csv(args.out)
    print(f"CSV dışa aktarıldı: {args.out}")


def cmd_import_sqlite(args) -> None:
    source_path = Path(args.sqlite_db)
    if not source_path.exists():
        raise SystemExit(f"SQLite veritabanı bulunamadı: {source_path}")

    target_db = GazetteDB(args.db)
    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row

    filters = []
    params = []
    if args.start:
        filters.append("date >= ?")
        params.append(args.start)
    if args.end:
        filters.append("date <= ?")
        params.append(args.end)

    sql = """
    SELECT date, source_url, item_url, title, category, institution, content,
           summary, content_hash, fetched_at, file_path
    FROM gazette_items
    """
    if filters:
        sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY date, id"
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)

    inserted = 0
    skipped = 0
    try:
        for row in source.execute(sql, tuple(params)):
            item = _row_to_gazette_item(row)
            if target_db._fetchone("SELECT id FROM gazette_items WHERE item_url = ? LIMIT 1", (item.item_url,)):
                skipped += 1
                continue
            if target_db.insert_item(item):
                inserted += 1
            else:
                skipped += 1
    finally:
        source.close()

    print(f"SQLite içe aktarma tamamlandı. Yeni kayıt: {inserted}, atlanan/var olan: {skipped}")


def cmd_show(args) -> None:
    db = GazetteDB(args.db)
    row = db.get_item(args.id)

    if not row:
        print("Kayıt bulunamadı.")
        return

    analyzer = GazetteAnalyzer(db)
    print("=" * 90)
    print(f"[{row['date']}] {row['title']}")
    print(f"Kategori: {row['category']}")
    print(f"Kurum: {row['institution'] or '-'}")
    print(f"URL: {row['item_url']}")

    print("\nÖzet:")
    print(row["summary"] or "-")

    print("\nAnahtar kelimeler:")
    for w, n in analyzer.document_keywords(row["content"] or "", 20):
        print(f"  - {w}: {n}")

    print("\nBenzer belgeler:")
    for sim in analyzer.similar_documents(args.id, limit=5):
        if "error" in sim:
            print("  -", sim["error"])
        else:
            print(f"  - {sim['similarity']}: [{sim['date']}] {sim['title']}")

    if args.full:
        print("\nTam metin:")
        print(row["content"] or "")


def cmd_serve(args) -> None:
    app = create_app(args.db)
    print(f"Web panel: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_rag_index(args) -> None:
    rag = RAGEngine(db_path=args.db, vector_db_path=args.vector_db)
    start_date = args.start
    end_date = args.end

    if args.days:
        end = dt.date.today()
        start = end - dt.timedelta(days=args.days - 1)
        start_date = start.isoformat()
        end_date = end.isoformat()

    count = rag.build_index(
        limit=args.limit,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"RAG indeksleme tamamlandı. Eklenen/güncellenen parça sayısı: {count}")


def cmd_rag_ask(args) -> None:
    rag = RAGEngine(db_path=args.db, vector_db_path=args.vector_db)
    answer = rag.answer_without_llm(question=args.question, top_k=args.top_k)
    print(answer)


def cmd_rag_llm(args) -> None:
    rag = RAGEngine(db_path=args.db, vector_db_path=args.vector_db)
    answer = rag.answer_with_llm(question=args.question, top_k=args.top_k, model=args.model)
    print(answer)
