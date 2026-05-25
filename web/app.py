# -*- coding: utf-8 -*-
"""
Flask web paneli.

Bu sürümde RAG + LLM sayfası da eklenmiştir.
"""

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEFAULT_DB
from core.analyzer import GazetteAnalyzer
from core.database import GazetteDB
from core.rag import RAGEngine


def create_app(db_path: str = DEFAULT_DB):
    app = Flask(__name__)
    db = GazetteDB(db_path)

    @app.route("/")
    def index():
        rows = db.list_items(limit=50)
        total = db.count_items()
        return render_template("index.html", app_name=APP_NAME, rows=rows, total=total)

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

    @app.route("/rag", methods=["GET", "POST"])
    def rag_page():
        question = ""
        answer = ""
        mode = "sources"
        error = ""

        if request.method == "POST":
            question = request.form.get("question", "").strip()
            mode = request.form.get("mode", "sources")

            if question:
                try:
                    rag = RAGEngine(db_path=db_path, vector_db_path="vector_db")
                    if mode == "llm":
                        answer = rag.answer_with_llm(question=question, top_k=5)
                    else:
                        answer = rag.answer_without_llm(question=question, top_k=5)
                except Exception as exc:
                    error = str(exc)

        return render_template(
            "rag_chat.html",
            app_name=APP_NAME,
            question=question,
            answer=answer,
            mode=mode,
            error=error,
        )

    @app.route("/api/stats")
    def api_stats():
        analyzer = GazetteAnalyzer(db)
        return jsonify(analyzer.build_report())

    return app
