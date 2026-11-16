# 10 · Glossary / Sözlük

| Term | Türkçe | Meaning in this project |
|---|---|---|
| Naive forecast | Naif tahmin | Tomorrow's price = today's price (no change). The price baseline. |
| MAE gain vs Naive | Naif'e göre MAE kazancı | 100 × (1 − MAE_model / MAE_naive). Target +5%. |
| MAPE | Ortalama mutlak yüzde hata | Mean of |forecast / final − 1|, in %. |
| QLIKE | QLIKE kaybı | Variance-forecast loss robust to noisy realised variance. |
| EWMA (λ = 0.94) | Üstel ağırlıklı hareketli ortalama | RiskMetrics variance reference. |
| HAR | HAR modeli | Heterogeneous autoregressive volatility: daily, weekly and monthly components. |
| GARCH(1,1) | GARCH | Conditional variance model; here zero mean and Student-t errors. |
| Rolling origin / walk-forward | Kayan başlangıç | Refit on past data only for each test period. |
| Inner / outer validation | İç / dış doğrulama | Selection on earlier blocks; scoring on the outer block. |
| Purge | Arındırma | Dropping overlapping labels between train and test. |
| Label maturity | Etiket olgunlaşması | A label can be used only after its horizon has passed. |
| Shrinkage weight | Küçültme ağırlığı | final = Naive + w·(model − Naive), w chosen on inner blocks. |
| Moving-block bootstrap | Hareketli blok bootstrap | Resampling blocks of 20/60 days to get CIs under autocorrelation. |
| Synthetic control | Sentetik kontrol | Data with a known injected signal, or a null, to test the pipeline. |
| Weak teacher | Zayıf öğretmen | A synthetic setup with a small known effect the learner should recover. |
| Ablation | Ablasyon | Removing or adding feature groups to measure their contribution. |
| Availability lag (L1/L2/L6) | Erişilebilirlik gecikmesi | A source enters at the first session after publication, or +1 / +5 sessions as a stress test. |
| Named contract | Adlandırılmış kontrat | A futures series pinned to one contract month, avoiding roll returns. |
| Preregistration | Ön kayıt | The pass rule written and committed before running. |
| Sealed holdout | Mühürlü test seti | Years kept unseen and opened once. |
| Blind test | Kör test | Data never looked at before the test (2000–2011 USDA forecasts). |
| Forward test / lock | İleri test / kilit | A forecast written before the outcome exists, with a timestamp in Git. |
| NDVI | Bitki örtüsü indeksi | (NIR − Red)/(NIR + Red); vegetation greenness. |
| MODIS MOD13Q1 / MYD13Q1 | MODIS | 250 m, 16-day vegetation index composites (Terra / Aqua). |
| Landsat C2 L2 | Landsat | 30 m surface reflectance, 16-day revisit per satellite. |
| CDL | Ekin Veri Katmanı | USDA Cropland Data Layer; class 2 = cotton. |
| FSA crop acreage | FSA ekili alan | Farm Service Agency planted and failed acres by county. |
| Abandonment | Terk edilen alan | 1 − harvested / planted; large in Texas drought years. |
| Ginnings | Çırçır | Bales processed by gins by a date; a late-season production signal. |
| Revision momentum | Revizyon momentumu | USDA's latest revision predicts the direction of its remaining error. |
| HGB | Gradient boosting | scikit-learn HistGradientBoostingRegressor. |
| Histogram CNN | Histogram CNN | CNN on band × time × bin histograms of cotton pixels (You et al. 2017). |
| Simplex ensemble | Simpleks karışım | Non-negative weights summing to 1, chosen on past out-of-sample errors. |
| Empirical Bayes shrinkage | Ampirik Bayes küçültme | State bias pulled toward 0 by the ratio of between- to within-state variance. |
| Kaggle kernel | Kaggle işi | A cloud CPU/GPU job run from the command line. |
| Receipt / ledger | Makbuz / defter | A hash-keyed record of each completed fit, so it is never repeated. |
