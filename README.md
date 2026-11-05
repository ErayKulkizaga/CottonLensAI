# CottonLensAI

CottonLensAI, ICE Cotton No. 2 günlük fiyat proxy'si üzerinde tahmin araştırmasını,
checksum doğrulanmış model paketleriyle çalışan bir FastAPI/Angular uygulamasıyla birleştirir.
Uygulama geçmiş tahminleri, model karşılaştırmalarını ve senaryo çıkarımını gösterir;
`live` ve `backtest` kayıtları ayrı tutulur.

## Güncel bilimsel durum — 5 Kasım 2026

**Fiyat için Naive, risk için EWMA referansları korunuyor. Bağımsız ileri tahmin üstünlüğü gösterilmedi.**
Fiyat araştırmasının kapıları en az %5 Naive-MAE kazancı, T+1 %53 / T+5 %55 yön
ve sekiz yıllık protokolde en az 6/8 kazanımdır. 2016–2023 ile incelenmiş 2024+
dönemleri yeniden kullanılan tarihsel kanıttır; bağımsız holdout değildir.

İki Windows toplama görevi 4 Kasım'da kapatıldı. Zamanında arşivlenmiş ilk ileri
kayıtlar korunuyor; görevlerin çalışmış olması model başarısı değildir.
Kaynak erişim saati, fiyat alanı ve kontrat kimliği sınırları [STATUS](docs/STATUS.md)'ta açıklanır.

| Çalışma | Ölçülen sonuç ve kapsam |
| --- | --- |
| [Fiyat-only koşullu medyan, T+1](docs/PRICE_ONLY_MEDIAN_RESULT_20261105.md) | 338 piyasa +4 sentetik fit; 2.006 origin, seçilmiş Naive-MAE kazancı −%0,0825; STOP. |
| [DXY/WTI zamanlama, T+1](docs/AVAILABILITY_CLOCK_RESULT_20261005.md) | 2.006 origin; seçilmiş Naive-MAE kazancı −%0,154. |
| [Bölgesel WASDE, T+1](docs/WASDE_REGIONAL_T1_RESULT_20261008.md) | 1.254 origin; numeric D0/D1 kazancı −%0,523 / −%0,011. |
| [Eşlenmiş fiyat-delta MAE/MSE](docs/PAIRED_PRICE_LOSS_RESULT_20261010.md) | 1.008 origin; MAE kolu MSE'den iyi, Naive kazancı −%0,488. |
| [CT own-level / DXY risk](docs/CT_LEVEL_DXY_CONTROL_20261030.md) | Ön kayıtlı ikincil own-level tarihsel koşulları geçti; birincil DXY belirsiz. |
| [Yakın dönem ve 2008–2009 risk](docs/STATUS.md) | Yakın dönemde 711 origin belirsiz; eski aynı-kontrat 252 origin'de seçilmiş model EWMA ile eşdeğer. |
| [Uçtan uca uygulama/araştırma denetimi](docs/END_TO_END_AUDIT.md) | Paket kimliği, hedef olgunluğu, native çıkarım ve yayın makbuzları denetlendi; kapsam sınırları belgede. |

Sentetik öğrenme kontrolleri piyasa performansı değildir. AMS yıllık kontrat
kimliği kusurlarından etkilenen sonuçlar temiz aynı-kontrat sinyal kanıtı sayılmaz.

## Uygulamayı başlatma

Docker Desktop'ın Linux motoru açıkken depo kökünde:

```bash
docker compose up --build
```

[Uygulama](http://localhost:8080) ve [API sözleşmesi](http://localhost:8080/api/v1/openapi.json).
`Development fixture` banner'ı sentetik gösterim verisini belirtir. Gerçek model
paketi `runtime/artifacts/current` içinde checksum ve içe aktarım kimliğiyle doğrulanır.
Backend eğitim paketleri veya CUDA içermez.

```bash
docker compose down
```

Bu komut kalıcı veritabanını silmez. [Belge indeksi](docs/README.md),
[backend](backend/pyproject.toml) ve [frontend](frontend/package.json) bağımlılıklarını açıklar.

## Araştırma ve doğrulama

Python 3.12 ve Node 24.15+ kullanın. ML için TensorFlow kurmadan ayrı CPU ortamı:

```bash
uv sync --project ml --only-group cpu --frozen
python ml/history.py validate
python ml/history.py check --family ridge --horizon 1
python ml/history.py check --query availability
```

CPU ortamını etkinleştirip `PYTHONPATH=ml/src:ml` (PowerShell: `$env:PYTHONPATH='ml/src;ml'`) ayarlayın:

```bash
python -m pytest -q ml/tests research/tests
```

Backend ve frontend kontrolleri kendi klasörlerinde:

```bash
cd backend
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
cd ../frontend
npm ci
npm run build
```

CI yalnız sentetik/sözleşme testleri yapar. Gerçek küçük CPU fit'leri açık izin,
tek süreç ve en fazla iki thread gerektirir. Sequence/GPU ve istatistiksel ARIMA
fit'leri Colab sınırında kalır. [Contributor kuralları](AGENTS.md) ve kilit dosyaları geçerlidir.

## Veri, modeller ve geri yükleme

Kod/test/belgeler Git'te; büyük kaynaklar, native modeller, scaler'lar, tahminler
ve ledger kanıtları checksum bağlı **private GitHub Releases** içinde tutulur.
[Deney sicili ve arşiv rehberi](research/README.md), [arşiv envanteri](research/archive-summary.json)
ve deneylerin kaynak/sonuç belgeleri ilgili Release'lere bağlantı verir.

```bash
gh release download evidence-20261005 --repo ErayKulkizaga/CottonLensAI --dir output/evidence-download
python ml/evidence_restore.py --archive-dir output/evidence-download --destination output/recovered --prefix output/full-year/experiments/research-availability-clock-pilot-v1/
```

Geri yükleme checksum doğrular, eğitim başlatmaz ve mevcut dosyaları değiştirmez.
Yeni paketlerin `research/storage.py` manifest biçimi için [arşiv rehberini](research/README.md) kullanın.
Bütün eski checkpoint'lerin mevcut olduğu iddia edilmez: legacy TF-placement ve
kısmi On-Call için bilinen model payload eksikleri korunmuş makbuzlarla belgelenmiştir.
Sırlar, `.env`, özel veritabanları ve kişisel yapılandırmalar arşiv kapsamına girmez.

## Sonraki araştırma

[Onaylanan Naive-MAE planı](docs/NAIVE_MAE_PLAN_20261105.md): önce eski kanıtları
ve Naive hatalarını doğrula; ardından yalnız fiyat geçmişiyle küçük koşullu medyan
ve Ridge kontrolünü ön kayıtlı bütçede karşılaştır. İlk sabit deney tamamlandı: [sonuç STOP](docs/PRICE_ONLY_MEDIAN_RESULT_20261105.md).

[GitHub kapsam teslimatı](docs/GITHUB_DELIVERY_20261105.md), [güncel durum](docs/STATUS.md),
[araştırma sözleşmesi](docs/TRADING_RESEARCH_CONTRACT_20261005.md) ve
[korunan README tarihçesi](docs/archive/README_HISTORY_20261105.md).

Sabit araştırma CLI'sı: `python ml/naive_mae.py --help`. İlk audit/prepare adımları
fit yapmaz; `pilot` yalnız açık CPU izni ve frozen kaynakla çalışır.
[Audit ve ön kayıt](docs/NAIVE_MAE_AUDIT_20261105.md).
