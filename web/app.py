# -*- coding: utf-8 -*-
"""
Flask web paneli.

Bu sürümde RAG + LLM sayfası da eklenmiştir.
"""

import contextlib
import datetime as dt
import io
import os
import threading
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEFAULT_DB
from core.analyzer import GazetteAnalyzer
from core.crawler import OfficialGazetteCrawler
from core.database import GazetteDB
from core.rag import RAGEngine
from core.utils import clean_extracted_text, date_range, parse_date
from core.vector_store import VectorStore


ADMIN_JOBS = {}
ADMIN_JOBS_LOCK = threading.Lock()
LATEST_ADMIN_JOB_ID = None


def create_app(db_path: str = DEFAULT_DB):
    app = Flask(__name__)
    app.jinja_env.filters["clean_text"] = clean_extracted_text
    db = GazetteDB(db_path)
    admin_jobs = ADMIN_JOBS
    admin_jobs_lock = ADMIN_JOBS_LOCK

    def _admin_authorized() -> bool:
        expected = os.getenv("CRAWL_ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN")
        if not expected:
            return False
        supplied = request.headers.get("X-Admin-Token") or request.args.get("token") or request.form.get("token")
        return (supplied or "").strip() == expected.strip()

    def _job_log(job_id: str, message: str) -> None:
        line = f"{dt.datetime.now().replace(microsecond=0).isoformat()} {message}"
        with admin_jobs_lock:
            job = admin_jobs.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(line)

    def _set_job(job_id: str, **values) -> None:
        with admin_jobs_lock:
            if job_id in admin_jobs:
                admin_jobs[job_id].update(values)

    def _public_job(job):
        if not job:
            return None
        return dict(job)

    def _log_captured_output(job_id: str, output: str) -> None:
        for line in output.splitlines():
            line = line.strip()
            if line:
                _job_log(job_id, line)

    def _date_counts(job_db: GazetteDB, start_date: str, end_date: str):
        return job_db._fetchall(
            """
            SELECT date, COUNT(*) AS belge_sayisi
            FROM gazette_items
            WHERE date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date
            """,
            (start_date, end_date),
        )

    def _run_crawl_job(job_id: str, start_date: str, end_date: str, should_index: bool) -> None:
        _set_job(job_id, status="running", started_at=dt.datetime.now().isoformat(timespec="seconds"))
        try:
            start = parse_date(start_date)
            end = parse_date(end_date)
            job_db = GazetteDB(db_path)
            crawler = OfficialGazetteCrawler(timeout=90, sleep=1.5, retries=2, max_request_seconds=240)

            found = 0
            inserted = 0
            skipped = 0
            errors = 0

            for day in date_range(start, end):
                _job_log(job_id, f"[TARA] {day}")
                try:
                    items = []
                    for attempt in range(3):
                        captured = io.StringIO()
                        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                            items = crawler.fetch_day(day)
                        _log_captured_output(job_id, captured.getvalue())
                        if items or attempt == 2:
                            break
                        _job_log(job_id, f"[TEKRAR] {day}: 0 kayıt, tekrar deneniyor ({attempt + 1}/2)")
                        threading.Event().wait(20)

                    found += len(items)
                    day_inserted = 0
                    day_skipped = 0
                    for item in items:
                        if job_db.insert_item(item):
                            inserted += 1
                            day_inserted += 1
                        else:
                            skipped += 1
                            day_skipped += 1
                    _job_log(job_id, f"[GUN OZET] {day}: bulunan={len(items)}, yeni={day_inserted}, atlanan={day_skipped}")
                except Exception as exc:
                    errors += 1
                    _job_log(job_id, f"[GUN HATA] {day}: {exc}")

            job_db.log_crawl(start.isoformat(), end.isoformat(), inserted, errors, notes=f"render_admin_job={job_id}; bulunan={found}; atlanan={skipped}")
            _job_log(job_id, f"[CRAWL] bulunan={found}, yeni={inserted}, atlanan={skipped}, hata={errors}")
            date_counts = _date_counts(job_db, start_date, end_date)
            for row in date_counts:
                _job_log(job_id, f"[DB] {row['date']}: {row['belge_sayisi']} belge")

            indexed_chunks = 0
            if should_index and found:
                _job_log(job_id, f"[QDRANT] indeksleme basliyor: {start_date} - {end_date}")
                rag = RAGEngine(db_path=db_path, vector_db_path="vector_db")
                indexed_chunks = rag.build_index(start_date=start_date, end_date=end_date)
                _job_log(job_id, f"[QDRANT] parca={indexed_chunks}")

            _set_job(
                job_id,
                status="completed" if found else "empty",
                finished_at=dt.datetime.now().isoformat(timespec="seconds"),
                found=found,
                inserted=inserted,
                skipped=skipped,
                errors=errors,
                indexed_chunks=indexed_chunks,
                date_counts=[dict(row) for row in date_counts],
            )
        except Exception as exc:
            _job_log(job_id, "[HATA] " + str(exc))
            _job_log(job_id, traceback.format_exc())
            _set_job(job_id, status="failed", error=str(exc), finished_at=dt.datetime.now().isoformat(timespec="seconds"))

    @app.route("/")
    def index():
        rows = db.list_items(limit=50)
        total = db.count_items()
        return render_template("index.html", app_name=APP_NAME, rows=rows, total=total)

    @app.context_processor
    def inject_data_coverage():
        dates = db.list_dates()
        if not dates:
            return {
                "data_coverage": {
                    "has_data": False,
                    "start": None,
                    "end": None,
                    "day_count": 0,
                    "document_count": 0,
                }
            }

        return {
            "data_coverage": {
                "has_data": True,
                "start": dates[-1][0],
                "end": dates[0][0],
                "day_count": len(dates),
                "document_count": sum(count for _, count in dates),
            }
        }

    @app.route("/search")
    def search_page():
        q = request.args.get("q", "").strip()
        results = db.search(q, limit=50) if q else []
        return render_template("search.html", app_name=APP_NAME, q=q, results=results)

    @app.route("/item/<int:item_id>")
    def item_page(item_id: int):
        row = db.get_item(item_id)
        if not row:
            return render_template("item.html", app_name=APP_NAME, row=None, keywords=[], similar=[])

        analyzer = GazetteAnalyzer(db)
        keywords = analyzer.document_keywords(row["content"] or "", 20)
        similar = analyzer.similar_documents(item_id, 5)
        return render_template("item.html", app_name=APP_NAME, row=row, keywords=keywords, similar=similar)

    @app.route("/stats")
    def stats_page():
        analyzer = GazetteAnalyzer(db)
        report = analyzer.build_report()
        return render_template("stats.html", app_name=APP_NAME, report=report)

    @app.route("/dates")
    def dates_page():
        dates = db.list_dates()
        total_days = len(dates)
        total_documents = sum(n for _, n in dates)
        return render_template(
            "dates.html",
            app_name=APP_NAME,
            dates=dates,
            total_days=total_days,
            total_documents=total_documents,
        )

    @app.route("/admin")
    def admin_page():
        return render_template("admin.html", app_name=APP_NAME)

    @app.route("/rag", methods=["GET", "POST"])
    def rag_page():
        question = ""
        answer = ""
        sources = []
        mode = "sources"
        error = ""

        if request.method == "POST":
            question = request.form.get("question", "").strip()
            mode = request.form.get("mode", "sources")

            if question:
                try:
                    rag = RAGEngine(db_path=db_path, vector_db_path="vector_db")
                    if mode == "llm":
                        sources = rag.prepare_sources(question=question, top_k=5)
                        answer = rag.answer_with_llm(question=question, top_k=5, sources=sources)
                    else:
                        sources = rag.prepare_sources(question=question, top_k=10)
                except Exception as exc:
                    error = str(exc)

        return render_template(
            "rag_chat.html",
            app_name=APP_NAME,
            question=question,
            answer=answer,
            sources=sources,
            mode=mode,
            error=error,
        )

    @app.route("/api/stats")
    def api_stats():
        analyzer = GazetteAnalyzer(db)
        return jsonify(analyzer.build_report())

    @app.route("/admin/crawl", methods=["POST"])
    def admin_crawl():
        if not _admin_authorized():
            return jsonify({"error": "unauthorized"}), 401

        payload = request.get_json(silent=True) or {}
        start_date = (request.form.get("start") or payload.get("start") or "").strip()
        end_date = (request.form.get("end") or payload.get("end") or "").strip()
        should_index_raw = str(request.form.get("index") or payload.get("index") or request.args.get("index") or "true").lower()
        should_index = should_index_raw not in {"0", "false", "no", "hayir", "hayır"}

        if not start_date or not end_date:
            return jsonify({"error": "start ve end zorunlu; format YYYY-MM-DD"}), 400

        try:
            start = parse_date(start_date)
            end = parse_date(end_date)
            if start > end:
                return jsonify({"error": "start tarihi end tarihinden buyuk olmamali"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        with admin_jobs_lock:
            global LATEST_ADMIN_JOB_ID
            for job in admin_jobs.values():
                if job.get("status") in {"queued", "running"}:
                    return jsonify({"error": "zaten calisan bir job var", "job": _public_job(job)}), 409

            job_id = uuid.uuid4().hex[:12]
            LATEST_ADMIN_JOB_ID = job_id
            admin_jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "start": start_date,
                "end": end_date,
                "index": should_index,
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "logs": [],
            }

        thread = threading.Thread(target=_run_crawl_job, args=(job_id, start_date, end_date, should_index), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id, "status_url": f"/admin/jobs/{job_id}"})

    @app.route("/admin/jobs/<job_id>")
    def admin_job_status(job_id: str):
        if not _admin_authorized():
            return jsonify({"error": "unauthorized"}), 401
        with admin_jobs_lock:
            job = admin_jobs.get(job_id)
            if not job:
                return jsonify({"error": "job bulunamadi"}), 404
            return jsonify(_public_job(job))

    @app.route("/admin/jobs/latest")
    def admin_latest_job_status():
        if not _admin_authorized():
            return jsonify({"error": "unauthorized"}), 401
        with admin_jobs_lock:
            if not LATEST_ADMIN_JOB_ID:
                return jsonify({"error": "job bulunamadi"}), 404
            job = admin_jobs.get(LATEST_ADMIN_JOB_ID)
            if not job:
                return jsonify({"error": "job bulunamadi"}), 404
            return jsonify(_public_job(job))

    @app.route("/admin/qdrant/dates")
    def admin_qdrant_dates():
        if not _admin_authorized():
            return jsonify({"error": "unauthorized"}), 401
        try:
            store = VectorStore()
            counts = store.date_counts()
            return jsonify({
                "date_count": len(counts),
                "chunk_count": sum(row["chunk_count"] for row in counts),
                "dates": counts,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/admin/delete-range", methods=["POST"])
    def admin_delete_range():
        if not _admin_authorized():
            return jsonify({"error": "unauthorized"}), 401

        payload = request.get_json(silent=True) or {}
        start_date = str(request.form.get("start") or payload.get("start") or "").strip()
        end_date = str(request.form.get("end") or payload.get("end") or "").strip()
        confirm = str(request.form.get("confirm") or payload.get("confirm") or "").strip()

        if not start_date or not end_date:
            return jsonify({"error": "start ve end zorunlu; format YYYY-MM-DD"}), 400

        try:
            start = parse_date(start_date)
            end = parse_date(end_date)
            if start > end:
                return jsonify({"error": "start tarihi end tarihinden buyuk olmamali"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        expected_confirm = f"DELETE {start_date} {end_date}"
        if confirm not in {"SIL", expected_confirm}:
            return jsonify({"error": "Onay icin confirm alanina 'SIL' yazilmali"}), 400

        end_exclusive = (end + dt.timedelta(days=1)).isoformat()

        try:
            job_db = GazetteDB(db_path)
            before_count = job_db.count_items_for_date_range(start_date, end_exclusive)
            deleted_db = job_db.delete_items_for_date_range(start_date, end_exclusive)

            qdrant_deleted = VectorStore().delete_date_range(start_date, end_exclusive)
            after_count = job_db.count_items_for_date_range(start_date, end_exclusive)

            return jsonify({
                "status": "completed",
                "start": start_date,
                "end": end_date,
                "end_exclusive": end_exclusive,
                "postgres_before": before_count,
                "postgres_deleted": deleted_db,
                "postgres_after": after_count,
                "qdrant": qdrant_deleted,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app
