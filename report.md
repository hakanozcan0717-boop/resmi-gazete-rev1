# Resmi Gazete Analiz ve RAG Sistemi - Proje Raporu

## 1. Projenin Amacı

Bu proje, Resmi Gazete içeriklerini otomatik olarak çekmek, saklamak, aramak, analiz etmek ve kullanıcının sorularına kaynaklı cevaplar üretebilmek için geliştirilmiş bir veri toplama ve RAG sistemidir.

Sistem temel olarak şu soruya cevap verir:

> Resmi Gazete'de yayımlanan karar, atama, yönetmelik, tebliğ ve benzeri içerikler nasıl daha hızlı aranabilir, analiz edilebilir ve LLM desteğiyle anlaşılır hale getirilebilir?

Proje; veri çekme, veritabanına kaydetme, metin temizleme, web paneli, Qdrant tabanlı vektör arama, OpenAI embedding, LLM destekli cevap üretme ve GitHub Actions ile otomatik güncelleme adımlarını bir araya getirir.

## 2. Genel Mimari

Proje Python tabanlıdır. Web arayüzü Flask ile çalışır. Lokal ortamda SQLite, canlı ortamda ise `DATABASE_URL` tanımlıysa PostgreSQL kullanılır. RAG tarafında Qdrant vektör veritabanı ve OpenAI embeddings kullanılır. LLM cevap üretimi için OpenAI-compatible Groq istemcisi desteklenir.

Ana bileşenler:

- `main.py`: Komut satırı giriş noktasıdır.
- `app.py`: Render/Gunicorn için Flask uygulaması girişidir.
- `core/crawler.py`: Resmi Gazete HTML/PDF içeriklerini çeker.
- `core/database.py`: SQLite/PostgreSQL veri katmanını yönetir.
- `core/rag.py`: RAG arama, hybrid sıralama, kaynak seçimi ve LLM prompt üretimini yönetir.
- `core/vector_store.py`: Qdrant ve OpenAI embeddings entegrasyonunu sağlar.
- `core/llm_client.py`: LLM cevap üretimi için OpenAI-compatible istemciyi kullanır.
- `core/utils.py`: Metin temizleme, mojibake düzeltme ve PDF kaynaklı bozuk karakter filtrelerini içerir.
- `web/app.py`: Flask web paneli ve admin veri çekme endpointlerini sağlar.
- `.github/workflows/daily_postgres_qdrant_update.yml`: GitHub Actions üzerinden günlük Render tetiklemesini yapar.

## 3. Veri Çekme Akışı

Sistem Resmi Gazete'nin günlük HTML adreslerini tarih bazlı olarak tarar. HTML içinde ilgili günün belge linkleri bulunursa her belge ayrı ayrı çekilir. Belge HTML ise metin doğrudan HTML'den çıkarılır. Belge PDF ise PDF indirilir ve metin çıkarma işlemi yapılır.

PDF metin çıkarma tarafında iki yol vardır:

- Öncelikli olarak PyMuPDF kullanılır.
- PyMuPDF çıktısı yetersiz veya bozuk görünürse pypdf yedek olarak denenir.

PDF metni kısa, bozuk veya sembol ağırlıklı görünüyorsa sistem aynı belgenin `.htm` veya `.html` eşlik eden sürümünü denemeye çalışır. Böylece PDF metin katmanı sorunlu olduğunda HTML kaynak üzerinden daha temiz içerik alma ihtimali korunur.

Veri çekme sonrasında her belge için şu alanlar kaydedilir:

- tarih
- kaynak URL
- belge URL
- başlık
- kategori
- kurum
- içerik
- özet
- içerik hash değeri
- çekilme zamanı
- varsa indirilen dosya yolu

## 4. Metin Temizleme ve Kalite Kontrol

Resmi Gazete PDF'lerinde ve bazı HTML kaynaklarında karakter kodlama problemleri oluşabildiği için projede özel temizlik fonksiyonları bulunur.

Temizlenen başlıca problemler:

- Mojibake karakterleri
- Yanlış decode edilmiş Türkçe karakterler
- PDF font encoding kaynaklı sembol yığınları
- Kontrol karakterleri
- Çok fazla `#`, `%`, `$`, kare veya anlamsız karakter içeren satırlar

Bu temizlik sadece veri kaydı sırasında değil; web görünümü, arama sonuçları, statik site çıktısı ve RAG kaynak/snippet alanlarında da kullanılır.

## 5. Arama ve Analiz Özellikleri

Web panel ve CLI üzerinden belge arama yapılabilir. Kullanıcılar tarih, başlık, kategori, kurum ve içerik üzerinden kayıtları inceleyebilir.

Analiz tarafında sistem:

- toplam belge sayısını,
- kategori dağılımını,
- en sık geçen kurumları,
- kelime frekanslarını,
- benzer belgeleri,
- belge özetlerini

üretebilir.

Bu özellikler özellikle Resmi Gazete içindeki yoğun metinleri daha hızlı taramak ve konu dağılımını görmek için kullanılır.

## 6. RAG ve LLM Entegrasyonu

RAG akışında veritabanındaki belgeler parçalara ayrılır. Bu parçalar OpenAI embeddings ile vektörlere dönüştürülür ve Qdrant koleksiyonuna kaydedilir.

Soru geldiğinde sistem:

1. Soruya göre Qdrant'tan en yakın parçaları bulur.
2. Yerel hybrid filtreleme ve niyet analiziyle sonuçları yeniden sıralar.
3. Kaynakları temizler.
4. Kullanıcı isterse LLM'e kaynaklı prompt gönderir.
5. LLM sadece verilen kaynaklara dayanarak Türkçe cevap üretir.

Sistem özellikle `kurumlara yapılan atamalar`, `hangi kuruma kim atanmış`, `atamaları listele` gibi sorular için yapılandırılmış cevap üretmeye çalışır. Uygun kaynak metin varsa `Tarih | Karar | Kişi | Kurum/Görev | Kaynak` yapısında liste oluşturur.

## 7. Web Panel

Flask tabanlı web panel şu işlevleri sağlar:

- Son belgeleri görüntüleme
- Metin arama
- Tarih bazlı belge kapsamını görme
- İstatistik ve analiz raporlarını inceleme
- Belge detayına gitme
- RAG + LLM soru-cevap ekranını kullanma
- Admin panelden veri çekme ve Qdrant indeksleme işlemlerini başlatma

Admin işlemleri için `CRAWL_ADMIN_TOKEN` veya `ADMIN_TOKEN` gerekir.

## 8. Otomatik Güncelleme

Proje GitHub Actions ile Render üzerinde çalışan admin endpointini tetikler. Günlük tetikleme Türkiye saatine göre ayarlanmıştır:

- 04:37
- 05:47

GitHub Actions tarafında cron UTC ile çalıştığı için workflow içinde bu saatlerin UTC karşılıkları kullanılır.

Workflow Render'daki `/admin/crawl` endpointini çağırır. Ağ gecikmesi veya Render soğuk başlangıcı gibi durumlara karşı retry ve timeout mantığı eklenmiştir.

## 9. Deploy Yapısı

Render deploy için temel dosyalar:

- `render.yaml`
- `Procfile`
- `requirements.txt`
- `runtime.txt`

Render ortamında gerekli başlıca environment değişkenleri:

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `GROQ_API_KEY`
- `CRAWL_ADMIN_TOKEN`

Veri çekme hızını ayarlamak için ayrıca şu değişkenler kullanılabilir:

- `CRAWL_SLEEP`
- `CRAWL_TIMEOUT`
- `CRAWL_RETRIES`
- `CRAWL_MAX_REQUEST_SECONDS`
- `CRAWL_EMPTY_DAY_SLEEP`

## 10. Güçlü Yanlar

Projenin güçlü tarafları şunlardır:

- Resmi Gazete verisini otomatik çekebilir.
- HTML ve PDF kaynaklarını birlikte değerlendirebilir.
- PyMuPDF ve pypdf ile PDF metin çıkarma dayanıklılığı artırılmıştır.
- Bozuk metinleri filtreleyen özel temizlik katmanı vardır.
- SQLite ve PostgreSQL desteği sayesinde hem lokal hem canlı ortamda çalışabilir.
- Qdrant + OpenAI embeddings ile semantik arama yapabilir.
- LLM ile kaynaklı Türkçe cevap üretebilir.
- Atama kararları için yapılandırılmış liste çıkarmaya çalışır.
- Render ve GitHub Actions ile otomatik güncelleme akışı kurulmuştur.

## 11. Sınırlamalar ve Dikkat Edilmesi Gerekenler

Bazı Resmi Gazete PDF'lerinde metin katmanı tamamen bozuk olabilir. Bu durumda PDF'den kişi, kurum veya karar detaylarını güvenilir şekilde çıkarmak mümkün olmayabilir. HTML eşlik eden dosya varsa sistem bunu kullanmaya çalışır.

Önceden bozuk metinle kaydedilmiş belgeler varsa sadece kodu düzeltmek yeterli değildir. İlgili tarih aralığının yeniden çekilmesi ve Qdrant indeksinin yenilenmesi gerekir.

LLM cevapları kaynak metin kalitesine bağlıdır. Kaynakta bilgi yoksa sistemin bilgi uydurmaması gerekir.

Render free plan kullanılıyorsa soğuk başlangıç ve uzun işlem süreleri görülebilir. Bu nedenle retry, timeout ve gece tetikleme stratejileri önemlidir.

## 12. Sonuç

Bu proje, Resmi Gazete verisini sadece depolayan basit bir scraper değildir. Veri çekme, metin temizleme, analiz, semantik arama, RAG ve LLM destekli cevap üretimini bir araya getiren bütünleşik bir sistemdir.

Güncel haliyle proje; günlük otomatik veri güncelleme, web panelden sorgulama, Qdrant indeksleme ve kaynaklı LLM cevapları üretme kabiliyetine sahiptir. En kritik geliştirme alanları ise bozuk PDF metinlerini daha iyi kurtarmak, eski kirli verileri yeniden indekslemek ve atama kararları gibi özel içerikler için daha güçlü yapılandırılmış çıkarım kuralları geliştirmektir.
