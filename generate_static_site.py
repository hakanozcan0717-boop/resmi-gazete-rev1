# -*- coding: utf-8 -*-

import html
import json
from pathlib import Path

from core.database import GazetteDB
from core.analyzer import GazetteAnalyzer


OUTPUT_DIR = Path("static_site")


def escape(value):
    return html.escape(str(value or ""))


def short(value, length=280):
    value = str(value or "").strip()
    if len(value) <= length:
        return value
    return value[: length - 3].rstrip() + "..."


def html_page(title, body):
    return f"""<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <title>{escape(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            background: #f4f6f8;
            color: #222;
        }}

        header {{
            background: #1f2937;
            color: white;
            padding: 22px 32px;
        }}

        .container {{
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 16px;
        }}

        .card {{
            background: white;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08);
            margin-bottom: 16px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            border-bottom: 1px solid #eee;
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }}

        th {{
            background: #f9fafb;
        }}

        .tag {{
            background: #e5e7eb;
            border-radius: 999px;
            padding: 4px 8px;
            display: inline-block;
            font-size: 13px;
        }}

        a {{
            color: #2563eb;
        }}

        .muted {{
            color: #666;
            font-size: 0.9em;
        }}

        .nav a {{
            margin-right: 12px;
        }}
    </style>
</head>

<body>
<header>
    <h1>{escape(title)}</h1>
    <p>GitHub Actions ile otomatik güncellenen Resmî Gazete raporu</p>
</header>

<div class="container">
    <div class="card nav">
        <a href="index.html">Ana sayfa</a>
        <a href="stats.html">Analiz</a>
        <a href="data.json">JSON veri</a>
    </div>

    {body}
</div>
</body>
</html>
"""


def generate_index(db):
    rows = db.list_items(limit=120)
    total = db.count_items()

    body = f"""
    <div class="card">
        <h2>Toplam Kayıt</h2>
        <p style="font-size:32px;">{total}</p>
        <p class="muted">Bu sayfa GitHub Actions tarafından otomatik güncellenir.</p>
    </div>

    <div class="card">
        <h2>Son Kayıtlar</h2>

        <table>
            <tr>
                <th>Tarih</th>
                <th>Başlık</th>
                <th>Kategori</th>
                <th>Kurum</th>
                <th>Kaynak</th>
            </tr>
    """

    for r in rows:
        body += f"""
            <tr>
                <td>{escape(r["date"])}</td>
                <td>
                    <strong>{escape(r["title"])}</strong><br>
                    <span class="muted">{escape(short(r["summary"], 260))}</span>
                </td>
                <td><span class="tag">{escape(r["category"] or "Diğer")}</span></td>
                <td>{escape(r["institution"] or "-")}</td>
                <td><a href="{escape(r["item_url"])}" target="_blank">Aç</a></td>
            </tr>
        """

    body += """
        </table>
    </div>
    """

    return html_page("Resmî Gazete Analiz Sistemi", body)


def generate_stats(db):
    analyzer = GazetteAnalyzer(db)
    report = analyzer.build_report()

    body = f"""
    <div class="card">
        <h2>Genel Bilgi</h2>
        <p>Toplam belge: <strong>{report["total_documents"]}</strong></p>
        <p>Rapor üretim zamanı: {escape(report.get("created_at", ""))}</p>
    </div>

    <div class="card">
        <h2>Kategori Dağılımı</h2>

        <table>
            <tr>
                <th>Kategori</th>
                <th>Adet</th>
            </tr>
    """

    for cat, n in report["category_distribution"]:
        body += f"""
            <tr>
                <td>{escape(cat)}</td>
                <td>{n}</td>
            </tr>
        """

    body += """
        </table>
    </div>

    <div class="card">
        <h2>En Sık Kelimeler</h2>

        <table>
            <tr>
                <th>Kelime</th>
                <th>Adet</th>
            </tr>
    """

    for word, n in report["top_words"][:60]:
        body += f"""
            <tr>
                <td>{escape(word)}</td>
                <td>{n}</td>
            </tr>
        """

    body += """
        </table>
    </div>
    """

    if isinstance(report.get("topic_clusters"), list):
        body += '<div class="card"><h2>Konu Kümeleri</h2>'

        for c in report["topic_clusters"]:
            keywords = ", ".join(c["keywords"])
            body += f"""
                <h3>Küme {c["cluster"]} - {c["size"]} belge</h3>
                <p>{escape(keywords)}</p>
            """

        body += "</div>"

    return html_page("Resmî Gazete Analiz Raporu", body), report


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    db = GazetteDB("resmi_gazete.db")

    index_html = generate_index(db)
    stats_html, report = generate_stats(db)

    (OUTPUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    (OUTPUT_DIR / "stats.html").write_text(stats_html, encoding="utf-8")
    (OUTPUT_DIR / "data.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Statik site üretildi: static_site/")


if __name__ == "__main__":
    main()