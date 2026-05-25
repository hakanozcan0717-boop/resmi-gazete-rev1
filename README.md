# Resmî Gazete Analiz ve Takip Sistemi

Bu proje, Resmî Gazete içeriklerini tarih aralığına göre çekmek, SQLite veritabanına kaydetmek, arama yapmak ve NLP / machine learning analizleri üretmek için hazırlanmıştır.

## Proje Yapısı

```text
resmi_gazete_projesi/
├── main.py
├── app.py
├── requirements.txt
├── Procfile
├── config/
│   └── settings.py
├── core/
│   ├── constants.py
│   ├── models.py
│   ├── utils.py
│   ├── http_client.py
│   ├── crawler.py
│   ├── database.py
│   ├── analyzer.py
│   └── commands.py
├── web/
│   ├── app.py
│   └── templates/
├── data/downloads/
└── exports/
```

## Kurulum

```bash
cd resmi_gazete_projesi
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

```bash
python main.py crawl --days 7
python main.py search --query "ihale"
python main.py analyze
python main.py serve
```

Web panel:

```text
http://127.0.0.1:5000
```

## Canlıya Alma

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
gunicorn app:app
```
