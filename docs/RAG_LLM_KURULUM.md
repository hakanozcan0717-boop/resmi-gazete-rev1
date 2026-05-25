# RAG + LLM Eklentisi Kurulum Rehberi

Bu paket mevcut `resmi_gazete_rev1` projesine RAG ve LLM özelliği ekler.

## 1. Dosyaları nereye kopyalayacaksın?

Bu paketteki dosyaları kendi proje klasörüne aynı yollarla kopyala.

Örneğin:

```text
core/rag.py              -> resmi_gazete_rev1/core/rag.py
core/vector_store.py     -> resmi_gazete_rev1/core/vector_store.py
core/llm_client.py       -> resmi_gazete_rev1/core/llm_client.py
core/commands.py         -> resmi_gazete_rev1/core/commands.py üzerine yaz
main.py                  -> resmi_gazete_rev1/main.py üzerine yaz
web/app.py               -> resmi_gazete_rev1/web/app.py üzerine yaz
web/templates/base.html  -> mevcut base.html üzerine yaz
web/templates/rag_chat.html -> yeni dosya olarak ekle
```

## 2. requirements.txt dosyasına eklenecekler

Mevcut `requirements.txt` dosyana şunları ekle:

```text
chromadb
sentence-transformers
openai
python-dotenv
```

Sonra çalıştır:

```bash
pip install -r requirements.txt
```

## 3. OpenAI API anahtarı

Proje ana klasöründe `.env` dosyası oluştur:

```text
OPENAI_API_KEY=buraya_api_anahtarini_yaz
OPENAI_MODEL=gpt-5.2
```

`.env.example` dosyasını kopyalayıp adını `.env` yapabilirsin.

## 4. Önce veri çek

```bash
python main.py crawl --days 7
```

## 5. RAG indeksini oluştur

```bash
python main.py rag-index
```

Bu işlemden sonra proje klasöründe `vector_db/` oluşur.

## 6. LLM olmadan RAG testi

```bash
python main.py rag-ask --question "İhale ile ilgili düzenlemeler nelerdir?"
```

Bu komut sadece ilgili kaynak parçalarını getirir.

## 7. LLM ile cevap üretme

```bash
python main.py rag-llm --question "İhale ile ilgili düzenlemeler nelerdir?"
```

Bu komut önce ilgili kaynakları bulur, sonra LLM'e gönderip cevap üretir.

## 8. Web panelden kullanma

```bash
python main.py serve
```

Tarayıcıda aç:

```text
http://127.0.0.1:5000/rag
```

## 9. Günlük otomasyon için

Eğer her gün veri çekiyorsan `.bat` dosyana şunu da ekle:

```bat
python main.py crawl --days 1
python main.py rag-index
```

Böylece yeni gelen Resmî Gazete verileri RAG sistemine de eklenir.
