# CottonLensAI

ABD pamuk rekoltesi ve pamuk vadeli fiyatı üzerine bir araştırma projesi. Projenin sorusu: USDA'nın aylık rekolte tahmininden daha isabetli bir tahmin yapılabilir mi? Yapılamıyorsa USDA nerede ve neden yanılıyor?

Her test, verisine bakılmadan önce yazılıp git'e kaydedildi. Başarısız sonuçlar da saklanıyor.

## Sonuçlar (17 Kasım 2026)

| İddia | Kanıt | Durum |
|---|---|---|
| Teksas hasat alanında USDA'dan daha isabetli | Hata %11,4 → %7,0 (2008–2025, p = 0,0005) | Kanıtlı; Teksas keşif eyaleti |
| USDA'nın Teksas hatası 2012'de verimden hasat alanına kaydı | Eylül alan hatası %3,8 → %14,9; Chow testi p = 0,0005 | Kanıtlı |
| USDA, kötü ürün durumuna gerekenin üçte ikisi kadar tepki veriyor | Oran 0,65–0,69; korelasyon −0,82 ile −0,88 | Kanıtlı |
| Kör sezonlarda üretim tahmini USDA'dan daha yakın | 23 sezon, 88/128 karşılaştırma, p = 0,007 | Kanıtlı; dönemleri birleştirme kararı sonradan alındı |
| USDA'nın bir sonraki revizyon yönü öngörülebilir | Kör pamukta %67, p = 0,012 (ön kayıtlı) | Eski dönemde var; 2012 sonrasında yok (%43) |
| Bugünkü üretim tahmini USDA'dan anlamlı ölçüde iyi | p = 0,12–0,19 | Hayır; alan ve verim hataları birbirini dengeliyor |
| Piyasada para kazandırır | T52 p = 0,30; T55 p = 0,082 | Hayır |
| Tahmin aralıkları kalibre | Gerçek zamanlı kapsama %83 / %93 | Evet, ama geniş (%80 aralık için ±%16) |

İleriye dönük test: 2026 tahminleri kilitlendi (ABD üretimi, Teksas, Oklahoma ve Kansas hasat alanı). Puanlama 10 Aralık 2026 ve Mayıs 2027'de yapılacak; ayrıntılar [FORWARD_LOCK_2026.md](docs/FORWARD_LOCK_2026.md) içinde.

Fiyat aşaması (Eylül–Kasım 2026, 32.934 kayıtlı model eğitimi) günlük fiyatta "fiyat değişmez" (naif) tahminini geçemedi. En iyi kazanç %0,22, hedef %5'ti. Bu aşamanın belgeleri [docs/archive/price-phase](docs/archive/price-phase/) altında.

## Nereden başlamalı

| Okuyucu | Belge |
|---|---|
| Yeni başlayan, kısa özet isteyen | [Araştırma paketi](docs/research_package/README.md) ve [sayılar](docs/research_package/02_NUMBERS_CHEATSHEET.md) |
| Hakem, araştırmacı | [Çalışma makalesi](docs/paper/WORKING_PAPER.md), [deney kaydı](docs/CROP_NOWCAST_TRIALS.md), [ön kayıtlar](docs/prereg/) |
| Mülakat | [Türkçe soru-cevap](docs/research_package/09_INTERVIEW_QA_TR.md), [portfolyo sayfası](docs/portfolio/cotton-usda-research.html) |
| Canlı sistem | [Monitör kararı (ADR)](docs/adr/0001-usda-forecast-monitor.md), [performans sayfası](docs/monitor/index.html) |
| Güncel plan | [PLAN_REVIEW5_20261117.md](docs/PLAN_REVIEW5_20261117.md) |

Bütün belgelerin dizini: [docs/README.md](docs/README.md).

## Depo yapısı

```
research/crop_nowcast/   rekolte araştırması: T39–T55 betikleri, CAL1, B2, kilitler
research/monitor/        canlı USDA rapor monitörü (model_v2, hash zincirli kayıt, publish)
research/evidence/       sonuç dosyaları; SHA-256 makbuzlu, değiştirilmez
research/tests/          rekolte ve monitör testleri
research/*.py            fiyat aşamasının denetim ve doğrulama betikleri (tarihsel)
ml/                      fiyat aşamasının ML paketi, deney sicili (ml/history.py), Colab notebook'ları
backend/, frontend/      fiyat aşamasının FastAPI + Angular gösterim uygulaması
docs/                    belgeler; dizini docs/README.md
.github/workflows/       CI ve usda-monitor zamanlanmış görevi
```

Büyük veri ve ara çıktılar `output/` klasöründe tutulur ve git'e girmez.

## Çalıştırma

Rekolte araştırmasının ana sonuçlarını (Teksas alan sonucu T51, T50 ve birleşik kör test dahil) yeniden üretmek; 10/10 sonuç eşleşir. Girdiler `output/` altında, parquet için `pyarrow` gerekir:

```bash
python research/reproduce.py
```

Monitör (USDA raporu, kayıt zinciri ve performans sayfası):

```bash
python research/monitor/monitor.py run 2026-12
python research/monitor/monitor.py verify
python research/monitor/publish.py
```

Zamanlanmış görev (`.github/workflows/usda-monitor.yml`) Ağustos–Aralık arasında her ayın 8'i ile 14'ü arasında çalışır. Rapor gelmezse ya da bir kontrol başarısız olursa otomatik olarak bir GitHub issue açar.

Testler:

```bash
PYTHONPATH=ml/src:ml python -m pytest -q research/tests
```

Fiyat aşamasının 18 testi (`test_ct_*`, `test_session_handoff`) parquet okuduğu için `pyarrow` ister (`python -m pip install pyarrow`); pyarrow ve `ml` PYTHONPATH'te olunca 409 testin hepsi geçer.

NASS QuickStats anahtarı `NASS_API_KEY` ortam değişkeninden ya da `~/.nass_api_key` dosyasından okunur. Anahtar repoya hiçbir zaman girmez.

### Fiyat aşaması uygulaması ve ML ortamı (tarihsel)

```bash
docker compose up --build
uv sync --project ml --only-group cpu --frozen
python ml/history.py check --family ridge --horizon 1
```

Bu komutlar fiyat aşamasının gösterim uygulamasını ve deney sicilini çalıştırır. Kurallar [AGENTS.md](AGENTS.md) içinde, eski README [docs/archive/README_HISTORY_20261105.md](docs/archive/README_HISTORY_20261105.md) içinde. Büyük kanıt paketleri (111 release, yaklaşık 5 GB) ayrı ve private `ErayKulkizaga/CottonLensAI-evidence` deposunun release'lerindedir; etiketler aynıdır. Bu repo public yapılmadan önce oraya taşındılar. Geri yükleme rehberi [research/README.md](research/README.md).

## Lisans ve veri

- Kod ve belgeler [MIT lisansı](LICENSE) altındadır.
- Repodaki veriler USDA (NASS, ESMIS, AMS, FAS), US Drought Monitor ve benzeri kamu kaynaklarından gelir; açık veri seti `docs/research_package/data/` altındadır. Üçüncü taraf fiyat verisinin (ICE, Yahoo) ham serileri repoda yoktur; yalnızca türetilmiş istatistikler ve bazı denetim kayıtlarında birkaç günlük kısa fiyat alıntıları vardır.
- Anahtarlar (`NASS_API_KEY` vb.) ortam değişkeninden ya da kullanıcı dizininden okunur ve repoya girmez.

## Kurallar

- Testler veriye bakılmadan önce yazılır (`docs/prereg/`); sonradan yapılan değişiklikler yalnızca tarihli veri ya da ayrıştırma düzeltmeleridir.
- `research/evidence/` altındaki dosyalar değiştirilmez; geçersiz koşular da saklanır.
- 2021 sonrası fiyat verisi mühürlüdür.
- Bu bir araştırma projesidir, alım satım sistemi değildir.
