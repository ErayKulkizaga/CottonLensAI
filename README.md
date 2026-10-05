# CottonLensAI

Pamuk piyasası için araştırma ve açıklanabilir tahmin uygulaması. Güncel ürün amacı **vadeli piyasada işlem yönü ve pozisyon kararını desteklemek**. Henüz maliyet sonrası işlem üstünlüğü doğrulanmadı; mevcut T+1/T+5 fiyat tahmininde Naive korunuyor.

## Nereden başlanır?

1. [Güncel durum ve kararlar](docs/STATUS.md).
2. [İşlem araştırmasının hedefi ve sınırları](docs/TRADING_RESEARCH_CONTRACT_20261005.md).
3. **Yeni deney önermeden önce** [deney sicilini](research/README.md) kontrol edin.

```bash
python ml/history.py validate
python ml/history.py check --family ridge --horizon 1
python ml/history.py check --feature nass --horizon 5
python ml/history.py check --query availability
```

Komut bağımlılık kurmadan çalışır ve dosya değiştirmez. `RELATED_EVIDENCE`, geçmişte ilgili çalışma bulunduğu anlamına gelir. Tam tarif ve dondurulmuş kapsam için `--proposal` kullanılabilir. Eşleşme bulunmaması bir fikrin hiç denenmediğini kanıtlamaz. Eksik tahminler ve yarım deneyler negatif kanıt sayılmaz.

## Depo düzeni

```text
backend/                 FastAPI, CPU inference, artifact sözleşmeleri ve testleri
frontend/                Angular uygulaması
ml/src/cottonlens_ml/     Veri, özellik, araştırma ve eski eğitim motorları
ml/tests/                Sentetik, zamanlama, kimlik ve sözleşme testleri
ml/notebooks/            İki güncel data/research workbench
ml/notebooks/archive/    Eski deneylerin notebook'ları; yeni çalışma girişleri değil
ml/scripts/              Mevcut kaynak toplama ve operasyon araçları
research/                Deney/fit sicili, düzeltme kanıtları, arşiv indeksi
research/evidence/       Küçük, checksum bağlı doğrulama ve sonuç kayıtları
docs/                    Güncel durum, karar ve kullanım belgeleri
docs/archive/            Tarihsel planlar ve yorumlar; güncel talimat değil
runtime/                 Yerel artifact/import alanı; büyük dosyalar Git'te tutulmaz
.github/                 Kod ve sözleşme CI kontrolleri
```

Python giriş araçları `ml/` altında kalır: `history.py`, `full_year_cpu.py`, `source_bundle.py`, `colab_setup.py` ve mevcut veri inceleme araçları. Kaynak/veri kimlikleri nedeniyle eski tarifler yeni kodla sessizce yeniden başlatılmaz.

## Ölçülmüş son durum

[Zamanlama deneyi](docs/AVAILABILITY_CLOCK_RESULT_20261005.md), 2016–2023'te her kol/ufuk için aynı 2.006 origin'de 676 küçük CPU fit tamamladı. DXY/WTI daha yeni barları fiilen kullandı; geçmiş Yahoo teslim saatleri **varsayımdır**.

| Sonuç | T+1 Naive'ye fiyat-MAE kazancı | T+5 kazancı |
|---|---:|---:|
| Kontrol, seçilmiş küçültme | −%0,3285 | −%0,8000 |
| Zamanlama müdahalesi, seçilmiş küçültme | −%0,1538 | −%1,3396 |

Bu tarif pratik hedefi geçmedi. Sonuç bütün modellerin veya bütün temel kaynakların sinyalsiz olduğunu kanıtlamaz. Tekrar incelenmiş 2016–2023 ve daha önce görülmüş 2024+ bağımsız holdout değildir.

Eski LSTM'nin 498 Naive / 439 LSTM origin karşılaştırması geçersizdi: aynı pencere için yeniden kurulan T+1 toplu kazanç **−%1,6961**, eski +%2,7647 iddiası geçersiz. Tam LSTM tahminleri bulunmadığından paired güven aralığı yok. TCN sentetik öğrenme kontrolünün yanlış `passed` yorumu ayrıca düzeltildi. [Düzeltme kayıtları](research/evidence/) eski artifact'leri değiştirmez.

[İlk işlem hedefi teslimatı](docs/TRADING_WASDE_DELIVERY_20261005.md), mevcut tahminlerde 2.006 T+1 origin'i yeni fit olmadan inceledi. Seçilmiş müdahalenin brüt proxy sonucu tek aktif yıldan geliyor; kontrol katkısı belirsiz ve gerçek giriş/maliyet doğrulanmış değil. Bölgesel WASDE bilgisi 95 rapor/26 alanlı aday pakete geri alındı, eğitim kabulü verilmedi. Sürümlü karar saati ve CLI/engine tazelik eşitliği düzeltildi; 690 eski dosyanın SHA-256'sı korundu. [Yeni payload Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/research-delivery-20261005) eski yedeğe ek kanıttır.

## Veriler, modeller ve tam kanıt yedeği

Büyük ham veriler, tahminler, checkpoint'ler, kaynak ZIP'leri ve ledger'lar [evidence-20261005 Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005) altında tutulur. [Arşiv açıklaması](research/README.md) ve `research/archive-summary.json`, mevcut dosyaların kapsamını, eksikleri, redaksiyonları ve SHA-256 değerlerini gösterir. Git geçmişi korunur; bütün eski eğitimin eksiksiz geri kazanıldığı iddia edilmez.

GitHub CLI ile indirme ve belirli bir deneyi ayrı dizine geri kurma:

```bash
gh release download evidence-20261005 --repo ErayKulkizaga/CottonLensAI --dir output/evidence-download
python ml/evidence_restore.py --archive-dir output/evidence-download --destination output/recovered --prefix output/full-year/experiments/research-availability-clock-pilot-v1/
```

Geri kurma arşiv ve dosya checksum'larını denetler, mevcut dosyaların üzerine yazmaz. Model yüklemez, eğitim başlatmaz ve canlı artifact'i değiştirmez. Gizli bilgiler ayıklanan kopyalar orijinal hash'leriyle eşit gösterilmez.

## Uygulamayı açma

Docker Desktop'ın Linux motoru açıkken:

```bash
docker compose up --build
```

Uygulama: <http://localhost:8080>. API: <http://localhost:8080/api/v1/openapi.json>.

`Development fixture` banner'ı sentetik gösterim verisi anlamına gelir. Gerçek artifact `runtime/artifacts/current` içinde checksum doğrulamasıyla yüklenir; eski artifact'in metrikleri güncel araştırma performansı gibi sunulmaz. `live` ve `backtest` kayıtları ayrıdır. Backend'e TensorFlow, MLflow, Jupyter veya CUDA kurulmaz.

```bash
docker compose down
```

Bu komut veritabanını silmez. Artifact kullanım/operasyon ayrıntıları [belge indeksinde](docs/README.md).

## Araştırma araçları

Güncel girişler [data workbench](ml/notebooks/cottonlens_data_workbench.ipynb) ve [research workbench](ml/notebooks/cottonlens_research_workbench.ipynb). Varsayılan akış durum/hazırlık kontrolüdür, eğitim değildir. Arşivdeki notebook'lar yalnız tarihsel protokolü incelemek içindir.

```bash
python ml/full_year_cpu.py --help
python ml/source_bundle.py --output output/new-source.zip
```

Yeni deney otomatik açılmaz. İzin verilen küçük CPU tarifleri ayrı CPU ortamında tek süreç ve en fazla iki thread kullanır; gerçek sequence/GPU ve istatistiksel ARIMA eğitimleri Colab sınırında kalır. CI gerçek piyasa eğitimi yapmaz. Kaynak kurulumları ve kimlik doğrulaması [AGENTS.md](AGENTS.md) ile mevcut kilit dosyalarına uyar. [USDA secret kurulumu](docs/USDA_COLAB_SETUP.md); gerçek anahtarlar depoya veya Release'e konmaz.

## Kontroller

```bash
python ml/history.py validate
# ML için mevcut ayrı CPU/test ortamında, PYTHONPATH=ml/src:ml (Windows: ml/src;ml)
python -m pytest -q ml/tests
cd backend
python -m pytest -q
python -m ruff check .
cd ../frontend
npm ci
npm run build
```

Frontend Node 24.15+ ister. Bağımlılık kimlikleri `ml/uv.lock`, `ml/constraints/`, backend manifesti ve npm lockfile'ında kayıtlıdır. Sentetik test veya build başarısı tahmin/işlem üstünlüğünün kanıtı değildir.

Eski `research-v2-tf-placement` için 7.331 fit ve kısmi on-call için 102 fit makbuzunun model payload kopyaları yerelde yok: toplam 22.299 dosya referansı. Makbuzlar/tahmin kanıtı korunur; bunlar yeniden kullanılabilir tam checkpoint gibi gösterilmez. Yeni zamanlama deneyinin 6.157 dosyası checksum doğrulamasıyla geri kurulmuştur. Kaybolan eski dosyalar yeni eğitimle yeniden üretilmedi.
