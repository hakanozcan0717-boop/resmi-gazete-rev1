# Resmi Gazete Analiz ve RAG Sistemi

Bu proje, Resmi Gazete içeriklerini tarih aralığına göre çeker, veritabanına kaydeder, arama ve analiz yapar, Qdrant + OpenAI embeddings ile RAG kaynak araması sağlar ve LLM ile Türkçe cevap üretir.

## Özellikler

- Resmi Gazete HTML/PDF içeriklerini günlük veya tarih aralığıyla çeker.
- SQLite ile lokal, `DATABASE_URL` varsa PostgreSQL ile canlı ortamda çalışır.
- Başlık, kategori, kurum, özet ve tam metin alanlarını kaydeder.
- Web panelinden listeleme, arama, tarih kapsamı, analiz ve detay görüntüleme sağlar.
- RAG + LLM sayfasında kaynak bulma ve LLM ile cevap üretme desteklenir.
- Qdrant üzerinde vektör indeks tutar.
- OpenAI embeddings kullanır.
- LLM cevabı için Groq OpenAI-compatible API istemcisi kullanır.
- Bozuk UTF-8/mojibake ve PDF font encoding kaynaklı çöp metinleri ekrana basmadan temizlemeye çalışır.
- Atama sorularında mümkünse `Tarih | Karar | Kişi | Kurum/Görev | Kaynak` tablosu üretir.
- Render admin endpointleriyle uzaktan crawl, indeksleme, job takibi ve tarih aralığı silme yapılabilir.
- GitHub Actions, Render üzerindeki günlük güncellemeyi Türkiye saatine göre tetikler.

## Proje Yapısı

```text
resmi_gazete_rev1/
├── main.py                         # CLI giriş noktası
├── app.py                          # Gunicorn/Render Flask giriş noktası
├── generate_static_site.py          # Statik HTML/JSON raporu üretir
├── requirements.txt
├── render.yaml
├── .github/workflows/
│   └── daily_postgres_qdrant_update.yml
├── config/
│   └── settings.py
├── core/
│   ├── analyzer.py                 # İstatistik, kelime frekansı, benzer belgeler
│   ├── commands.py                 # CLI komut implementasyonları
│   ├── crawler.py                  # Resmi Gazete crawler
│   ├── database.py                 # SQLite/PostgreSQL veri katmanı
│   ├── http_client.py              # HTTP retry/timeout istemcisi
│   ├── llm_client.py               # Groq/OpenAI-compatible LLM istemcisi
│   ├── rag.py                      # RAG, hybrid filtreleme, prompt ve atama listesi
│   ├── utils.py                    # Metin temizleme, mojibake/PDF çöp filtreleri
│   └── vector_store.py             # Qdrant + OpenAI embeddings
├── web/
│   ├── app.py                      # Flask web paneli ve admin API
│   └── templates/
├── data/downloads/                 # İndirilen PDF dosyaları
├── exports/
└── static_site/
```

## Kurulum

Windows PowerShell:

```powershell
cd C:\Users\DELL\Desktop\projects\resmi_gazete_rev1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ortam Değişkenleri

Lokal kullanım için proje kökünde `.env` dosyası kullanılabilir. Bu dosya GitHub'a gönderilmemelidir.

```text
# Lokal veya canlı PostgreSQL kullanmak için
DATABASE_URL=postgresql://...

# RAG embedding için
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant için
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=resmi_gazete

# LLM cevabı için Groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant

# Admin crawl endpointleri için
CRAWL_ADMIN_TOKEN=...
```

Notlar:

- `DATABASE_URL` yoksa sistem SQLite dosyası olan `resmi_gazete.db` ile çalışır.
- `QDRANT_URL`, `QDRANT_API_KEY` ve `OPENAI_API_KEY` olmadan RAG indeksleme/arama çalışmaz.
- `GROQ_API_KEY` olmadan LLM ile cevap üretme çalışmaz; sadece kaynak getirme kullanılabilir.

## CLI Kullanımı

Yardım:

```powershell
venv\Scripts\python.exe main.py --help
```

Veri çekme:

```powershell
venv\Scripts\python.exe main.py crawl --days 2
venv\Scripts\python.exe main.py crawl --start 2026-05-31 --end 2026-06-01
```

Günlük güncelleme ve Qdrant indeksleme:

```powershell
venv\Scripts\python.exe main.py daily-update --days 2
```

Arama:

```powershell
venv\Scripts\python.exe main.py search --query "ihale"
```

Analiz:

```powershell
venv\Scripts\python.exe main.py analyze
venv\Scripts\python.exe main.py analyze --json-out exports/gunluk_analiz.json
```

RAG indeks oluşturma:

```powershell
venv\Scripts\python.exe main.py rag-index
venv\Scripts\python.exe main.py rag-index --start 2026-05-31 --end 2026-06-01
```

RAG kaynak arama:

```powershell
venv\Scripts\python.exe main.py rag-ask --question "kurumlara yapılan atamaları göster"
```

RAG + LLM cevap:

```powershell
venv\Scripts\python.exe main.py rag-llm --question "vergiyle ilgili kararları özetle"
```

Web panel:

```powershell
venv\Scripts\python.exe main.py serve
```

Adres:

```text
http://127.0.0.1:5000
```

## Web Panel

Başlıca sayfalar:

- `/` son kayıtlar ve veri kapsamı
- `/search` metin arama
- `/dates` tarih bazlı belge sayıları
- `/stats` analiz raporu
- `/item/<id>` belge detayı
- `/rag` RAG + LLM Soru-Cevap
- `/admin` Render/admin veri çekme paneli

Admin endpointleri `CRAWL_ADMIN_TOKEN` veya `ADMIN_TOKEN` ister.

## RAG + LLM Davranışı

RAG akışı:

1. SQLite/PostgreSQL kayıtları parçalara bölünür.
2. OpenAI embeddings ile vektörler üretilir.
3. Qdrant koleksiyonuna upsert edilir.
4. Soru geldiğinde Qdrant'tan aday parçalar alınır.
5. `core/rag.py` içindeki hybrid filtre ve intent mantığı sonuçları yeniden sıralar.
6. İstenirse LLM'e temizlenmiş kaynaklarla prompt gönderilir.

Atama soruları için sistem ayrıca şunu dener:

- Soru `kurumlara yapılan atamalar`, `hangi kuruma kim atanmış`, `atamaları listele` gibi ise yapılandırılmış cevap üretir.
- Kaynak metinde açıkça `... görevine Ahmet Yılmaz atanmıştır` benzeri satırlar varsa tablo çıkarır.
- PDF metni bozuksa veya kişi/kurum bilgisi metin katmanında yoksa bunu açıkça belirtir; bilgi uydurmaz.

## Metin Temizleme

`core/utils.py` iki tür problemi azaltır:

- Mojibake: `ResmÃ®`, `iÃ§erik`, `ÅŸ` gibi yanlış decode edilmiş metinleri onarır.
- PDF font encoding çöpü: `#!#`, `□□□`, `%%%%`, kontrol karakterleri ve sembol ağırlıklı satırları filtreler.

Bu temizlik yeni crawl kayıtlarında, web görünümlerinde, statik site üretiminde ve RAG kaynak/snippet alanlarında kullanılır.

## Render Deploy

`render.yaml` temel ayarları içerir.

Build command:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Render environment değişkenlerinde en az şunlar gerekir:

```text
DATABASE_URL
OPENAI_API_KEY
QDRANT_URL
QDRANT_API_KEY
CRAWL_ADMIN_TOKEN
GROQ_API_KEY
```

## GitHub Actions Günlük Tetikleme

Workflow:

```text
.github/workflows/daily_postgres_qdrant_update.yml
```

GitHub cron UTC çalışır. Mevcut schedule Türkiye saatiyle:

```text
04:37 Türkiye saati
05:47 Türkiye saati
```

Workflow içindeki UTC karşılığı:

```yaml
schedule:
  - cron: "37 1 * * *"
  - cron: "47 2 * * *"
```

Workflow, Render'daki `/admin/crawl` endpointini çağırır. Timeout/ağ gecikmelerinde 120 saniye bekler ve 4 denemeye kadar retry yapar.

GitHub Secrets:

```text
RENDER_APP_URL
CRAWL_ADMIN_TOKEN
```

Manuel çalıştırma için GitHub Actions ekranındaki `workflow_dispatch` kullanılabilir. İsteğe bağlı inputlar:

- `days`
- `crawl_start`
- `crawl_end`

## Statik Site Üretimi

```powershell
venv\Scripts\python.exe generate_static_site.py
```

Çıktılar:

```text
static_site/index.html
static_site/stats.html
static_site/data.json
```

## Bilinen Sınırlamalar

- Bazı Resmi Gazete PDF'lerinde metin katmanı bozuk olabilir. Bu durumda kişi/kurum gibi ayrıntılar çıkarılamaz.
- Qdrant eski kirli payload ile indekslendiyse, temizleme kodu görüntüde yardımcı olur; en doğru sonuç için ilgili tarih aralığını silip yeniden indekslemek gerekir.
- LLM cevapları sadece verilen kaynaklara dayanacak şekilde promptlanır; kaynakta olmayan bilgi üretilmemelidir.
- Render free plan soğuk başlangıç veya uzun işlem nedeniyle yavaş cevap verebilir; GitHub workflow retry mekanizması bu yüzden eklenmiştir.
