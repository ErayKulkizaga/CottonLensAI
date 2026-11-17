# Belge dizini

Belgeler üç gruba ayrılır: **güncel** (rekolte araştırması ve canlı sistem), **ön kayıtlar** ve **arşiv** (eski planlar ve fiyat aşaması). Arşivdeki belgeler geçmişin kaydıdır, güncel talimat değildir.

## Güncel
| Belge | İçerik |
|---|---|
| [CROP_NOWCAST_TRIALS.md](CROP_NOWCAST_TRIALS.md) | Rekolte araştırmasının deney kaydı (T01–T55), başarısızlar dahil |
| [FORWARD_LOCK_2026.md](FORWARD_LOCK_2026.md) | 2026 için kilitlenen tahminler ve puanlama tarihleri |
| [PLAN_REVIEW5_20261117.md](PLAN_REVIEW5_20261117.md) | Güncel plan ve durumu |
| [paper/WORKING_PAPER.md](paper/WORKING_PAPER.md) | Çalışma makalesi taslağı (İngilizce) |
| [research_package/](research_package/README.md) | Araştırma paketi: özet, sayılar, veri kaynakları, protokol, başarısızlıklar, Türkçe mülakat soru-cevapları, sözlük, açık veri seti |
| [adr/0001-usda-forecast-monitor.md](adr/0001-usda-forecast-monitor.md) | Canlı monitörün tasarım kararları |
| [monitor/index.html](monitor/index.html) | Monitörün performans geçmişi; iş akışı tarafından üretilir |
| [portfolio/](portfolio/) | İngilizce portfolyo sayfası (`cotton-usda-research.html`) ve Türkçe sunum (`cottonlens-sunum.html`) |

## Ön kayıtlar ([prereg/](prereg/))
Her dosya, testin kuralını verisine bakılmadan önce tarihli olarak kaydeder; sonuç bölümü sonradan eklenir.

| Test | Konu | Sonuç |
|---|---|---|
| [SEALED_TEST_T39](prereg/SEALED_TEST_T39_PREREG_20261115.md) | Mühürlü 2022–2024 testi | Başarısız |
| [LINK1](prereg/LINK1_EVENT_STUDY_PREREG_20261115.md) | Rapor günü fiyat tepkisi | Fiyat tepki veriyor (1,42×), revizyon yönü sinyal değil |
| [T46](prereg/T46_CORN_SOY_PREREG_20261116.md) | Momentum kuralı mısır ve soyada | Başarısız |
| [T47](prereg/T47_BLIND_1985_1999_PREREG_20261116.md) | Kör test 1985–1999 | Sabit katsayılarla başarısız |
| [T48](prereg/T48_MECHANISM_PREREG_20261116.md) | Hata mekanizması: alan mı, verim mi | 2/4 hipotez |
| [T49](prereg/T49_ANCHORING_PREREG_20261116.md) | Hasat alanında sabitlenme | Teksas'ta var; OK/KS p = 0,056 |
| [T50](prereg/T50_REVISION_DIRECTION_PREREG_20261116.md) | Sonraki revizyonun yönü | Geçti (%67) |
| [T51](prereg/T51_AREA_PRODUCTION_PREREG_20261116.md) | Alan düzeltmesi, aralıklar, üretim | Alanda geçti, üretimde başarısız |
| [T52](prereg/T52_MARKET_PREREG_20261116.md) | Piyasa testi 2010–2021 | Başarısız |
| [T53](prereg/T53_AREA_PRODUCTION_LONG_PREREG_20261116.md) | Uzun tarihte üretim | Başarısız |
| [T54](prereg/T54_SORGHUM_PREREG_20261116.md) | Sorgum | İşaret var, düzeltme yok |
| [T55](prereg/T55_MARKET_2000_2009_PREREG_20261117.md) | Piyasa testi 2000–2009 | Başarısız |
| [CAL1](prereg/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md) | Tahmin aralığı kalibrasyonu | Yalnızca "2 kat" yöntemi kalibre |

T39 öncesi denemelerin kuralları [CROP_NOWCAST_TRIALS.md](CROP_NOWCAST_TRIALS.md) içinde.

## Arşiv ([archive/](archive/))
| Klasör | İçerik |
|---|---|
| [archive/phase2-notes/](archive/phase2-notes/) | Rekolte aşamasının eski planları, ara özetleri ve devir notları (13–16 Kasım) |
| [archive/price-phase/](archive/price-phase/) | Fiyat aşamasının 101 belgesi (Ekim–Kasım 2026): protokoller, sonuçlar, denetimler, `STATUS.md` |
| [archive/](archive/README.md) | En erken planlar, model kartı ve demo notları (Eylül–Ekim 2026) |

Fiyat aşamasının özeti: [research_package/03_PHASE1_PRICE_AND_RISK.md](research_package/03_PHASE1_PRICE_AND_RISK.md).
