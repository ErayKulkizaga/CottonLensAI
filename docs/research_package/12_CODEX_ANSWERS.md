# 12 · Codex’in Phase 1 yanıtları

16 Kasım 2026. İnceleme, Claude çalışma ağacındaki `1247455` ile `59177d7` paket sürümünü, araştırma worktree’sindeki `445d1fe` kayıtlarını ve yerel arşivleri karşılaştırır. İncelenen 03 ve 04’ün içerikleri iki paket sürümünde aynıdır. Aksi belirtilmedikçe `docs/`, `research/` ve `ml/` yolları `445d1fe` ağacına; `output/` ve `CottonLensAI Colab/` yolları ana CottonLensAI checkout’una; `docs/research_package/` yolları Claude checkout’una göredir. `output/claude-audit/drive/` aşağıda adı geçen özgün Drive dosyalarının bulunan yerel kopyasıdır. Arşiv kapsamı her sonuçta ayrıca belirtilir.

Konumlar: ana checkout `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI`; araştırma worktree’si `C:/Users/Eray/.codex/worktrees/naive-mae-20261105/CottonLensAI`; ikinci worktree `C:/Users/Eray/.codex/worktrees/repository-curation/CottonLensAI`; Claude checkout’u `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI-claude-review`.

Her soruda **kesin** arşiv, kod, makbuz veya yeniden sayımla dayanaklandırılan ana yanıttır. Ana sorunun önemli bir parçası açık kaldığında **bilmiyorum** kullandım ve doğrulanan alt bulguları yazdım. Yerel Codex oturum dizinlerinde ilgili Kasım oturumu bulundu. Eylül çalışmaları veya eşiklerin ilk konuşması için eşleşen oturum bulunamadı. Hafızadan sayı kullanılmadı.

## B · LSTM

### 4. V1 piyasa LSTM’inin mimarisi

**Cevap** — Girdi 60 tutulmuş, feature’ları eksiksiz gözlem × 24 feature. Eksik tarihler nedeniyle bu pencere otomatik olarak 60 ardışık borsa seansı demek değildir. Bir Keras LSTM katmanı (32 veya 64 hücre, varsayılan `tanh` / `sigmoid`, katman içi dropout 0 ve recurrent dropout 0), ardından ayrı `Dropout` (0,10 veya 0,20), Dense(32, ReLU), Dense(2, lineer, `returns`). Çıktılar aynı ağdan T+1 ve T+5 log-getirileri. Adam, nominal öğrenme oranı 0,001, batch 64, en çok 100 epoch. Kayıp iki standartlaştırılmış hedef üzerinde MAE. Early stopping ve en iyi model kaydı `val_loss` izler, patience 10 ve en iyi ağırlıkları geri yükleme açık. Seed 42 eğitim fonksiyonu girişinde ayarlanır. `StandardScaler` 24 girdiyi train satırlarından, ayrı scaler iki hedef sütununu train sekanslarından öğrenir. Getiri tahmini ters ölçeklenir ve fiyat hesabında kullanılır.

24 feature: pamuk log-getiri 1/5/10/20, momentum 5/20, SMA oranı 5/20 ve 20/50, volatilite 5/20, volatilite rejimi 20/60, gün içi range, hacim değişimi ve 20 günlük hacim z-skoru, DXY ve WTI log-getirileri 1/5/20, pamuk–DXY ve pamuk–WTI 60 gözlem korelasyonları, ay sin/cos. DXY/WTI pamuk takvimine hizalanıp bir gözlem geciktirilir. `FEATURE_NAMES` listesinde CFTC yoktur.

**Kesinlik** — kesin.

**Kaynak** — `c33d6ae:ml/src/cottonlens_ml/training.py:211–349`, `config.py:4–29`, `features.py:13–69` (aynı commit ve paket dizini). `output/claude-audit/drive/checkpoints/lstm-2bb3164d6e23-32-20.keras` içindeki `config.json` 32 hücre, 0,20 harici dropout, iki lineer çıktı, Adam ve float32 olarak 0,001000000047 öğrenme oranını doğruluyor. Keras ZIP CRC kontrolü geçti.

**Claude nereye bakmalı** — Mimariyi commit’ten ve saklı checkpoint’in `config.json` dosyasından okuyabilir. Modeli deserialize etmeye gerek yok.

### 5. Dört konfigürasyon

**Cevap** — (hücre, ayrı dropout) = (32, 0,10), (32, 0,20), (64, 0,10), (64, 0,20). Dört outer fold’un her birinde dört iki-çıktılı aday, ayrıca geniş train/validation bölmesinde aynı dört aday. Raporun T+1 ve T+5 altında aynı denemeleri iki kez göstermesi iki ayrı eğitim bütçesi oluşturmaz.

**Kesinlik** — kesin.

**Kaynak** — `c33d6ae:ml/src/cottonlens_ml/training.py:240–257,320–348`, `walkforward.py:155–166`, `pipeline.py:28–36` (aynı paket). `output/claude-audit/drive/reports/cottonlens-results-v20260923-0652.txt:131–140,142–255`.

**Claude nereye bakmalı** — Dört fold’un bloklarını ve final validation listesini yan yana okuyabilir.

### 6. Epoch 1 ve kayıp eğrileri

**Cevap** — Epoch 1, dört outer fold dışındaki final tuning split’inde 32 hücre / dropout 0,20 adayının minimum `val_loss` değeridir. Train 17 Mart 2022’de biter; validation 25 Mart 2022–10 Haziran 2024. Ölçüm iki ölçeklenmiş log-getiri hedefinin MAE kaybı. Epoch 1 train/val 0,73125 / 0,92163, epoch 11’de 0,63588 / 0,97850. Fold’ların epoch seçimi farklı olabilir. Örneğin v1 fold 1’de bu aday en iyi epoch 2’de, düzeltilmiş R2 fold 4’te seçilen 64 / 0,20 adayının en iyi epoch’u 3. V1 eğrisi 11 nokta halinde rapor JSON’unda ve metninde, ayrıca candidate `*.history.json` dosyalarında saklı.

**Kesinlik** — kesin.

**Kaynak** — `output/claude-audit/drive/reports/cottonlens-results-v20260923-0652.txt:29,131–165,256–268`, eş adlı JSON’da `metrics` içindeki LSTM `training_history`; `output/claude-audit/drive/checkpoints/lstm-2bb3164d6e23-32-20.history.json`. R2 için `docs/archive/DAY1_R2_EVIDENCE_REVIEW.md:9–23`.

**Claude nereye bakmalı** — V1 eğrisi için yukarıdaki metin/JSON; R2 eğrisi için R2 belgesinde adı verilen hazır olma kimliği altındaki fold 4 geçmiş dosyası. R2 raporunda boş eğri görünmesi, audit refit’te tarihçe alanının temizlenmesinden kaynaklanır.

### 7. DG-008 ve kurtarma

**Cevap** — **Önemli düzeltme:** DG-008, 22 Eylül tarihli `v20260922-1450` ve `13761b9` koduna ait. 23 Eylül tarihli `v20260923-0652` / `c33d6ae` dört-fold çalışması ayrı. Önceki yanıtım bu iki çalışmayı yeterince ayırmıyordu.

22 Eylül release ZIP’i `output/claude-audit/drive/artifacts/releases/cottonlens-model-v20260922-1450.zip` altında bulundu. Dış SHA-256 eşleşti, ZIP CRC geçti. ZIP içindeki `metrics.json`, `forecasts.parquet` ve `manifest.json` hash’leri `research/evidence/legacy-lstm-correction.json` içindeki özgün artifact hash’leriyle aynı. ZIP’te iki deneysel XGBoost modeli var. Kayıp LSTM tahmin vektörü ve LSTM model payload’u ZIP içinde ayrı bir dosya olarak bulunmadı. Forecast Parquet bellekte okundu. 998 satır var, 996 backtest ve 2 live, her ufuk için 498 backtest + 1 live. Dosya model adı sütunu taşımıyor. Manifest iki ufukta birincil tahmini Naive olarak işaretliyor, dolayısıyla bu kayıt LSTM’nin 439 satırlık ayrı tahmin vektörünü sağlamıyor. DG-008 düzeltme makbuzu da tahmin payload’unu yok gösterir.

Eski karşılaştırma 498 Naive ve 439 LSTM origin’ini kullanıyordu. T+1 için +%2,764668 eski kıyas geçersiz. Aynı pencereye yeniden kurulan Naive MAE 0,659772 ve saklı LSTM aggregate MAE 0,670963 ile fark −%1,696141. T+5 aynı pencere farkı −%4,813858. Bunlar kayıtlı LSTM aggregate değerleri ile yeniden hesaplanan aynı-pencere Naive karşılaştırması. LSTM satırları bulunmadığı için paired CI ve satır bazlı doğrulama yok.

22 Eylül ve 23 Eylül mimarileri de ayrıdır. 22 Eylül kodunda altı aday var. (32, .10), (48, .15), (64, .20), (64, .30), (96, .20), (96, .30). Q4–Q6’daki dört aday ve hedef standardizasyonu 23 Eylül kodunun özellikleridir. Aynı seed ve commit, eksik geçmiş tahminlerin özgün byte’larını garanti etmez.

**Kesinlik** — bilmiyorum. Özgün 22 Eylül LSTM satır vektörünün yedeği henüz doğrulanmadı. ZIP ve düzeltme makbuzunun hash eşleşmesi ve karşılaştırma sayıları kesin.

**Kaynak** — `research/evidence/legacy-lstm-correction.json:2–40`, `docs/archive/price-phase/AVAILABILITY_CLOCK_RESULT_20261005.md:75–84`, `research/data-gpu-defects.json:130–139`; `13761b9:ml/src/cottonlens_ml/training.py`; yerel 22 ve 23 Eylül ZIP’leri, `output/claude-audit/drive/artifacts/releases/`.

**Claude nereye bakmalı** — 22 Eylül ZIP’i ve özgün Colab tracking geçmişini incele. Forecast Parquet’te model adı sütunu yok ve manifest birincil tahmini Naive olarak gösteriyor. 23 Eylül checkpoint’lerini 22 Eylül fit state’i olarak kullanma. Colab arşivinde ilgili raporlar, deneyler ve dosya adları hedefli tarandı. Bulunmayan vektöre ilişkin sonuç bu taramanın kapsamıyla sınırlı.

### 8. Round C kararı

**Cevap** — Round C, her aile **ve her ufuk** için 48 MLP/LSTM/TCN adayı ve 20/60/120 pencereler olarak tasarlandı. Round A’nın zayıf dış sonuçlarından sonra yön/sequence/ensemble koşullu hipotez olarak bırakıldı. 28 Eylül reset’i mevcut bilgide genişlemeyi durdurup önce kabul edilebilir kaynak ve eşlenmiş ablation istemişti. İncelenen fit defteri bu aileler için yalnız birer sentetik bilinen-sinyal kontrolü içeriyor. 23 Eylül legacy piyasa LSTM’i ayrı çalışma.

**Kesinlik** — kesin.

**Kaynak** — `docs/archive/CONTINUOUS_RESEARCH.md:11–20`, `docs/archive/RESEARCH_RESET_20260928.md:5–18,24–26,36–43`, `ml/src/cottonlens_ml/research/search.py:6–9,25–40`, `research/trials.json`.

**Claude nereye bakmalı** — Tasarım ve devam kararı için iki arşiv belgesi; yürütülen kapsam için aynı deneyin `decisions/` ve ledger kayıtları.

## E · Own-level HAR ileri testi

### 16. Şu an çalışıyor mu?

**Cevap** — 16 Kasım kontrolünde kapalı. `CottonLens-Risk-Daily-v1` ve ayrı eski `CottonLens-Observed-Archive-v2` Windows görevleri `Disabled`. Günlük HAR görevi 03:05 ve 03:20 Türkiye saati için kaynak alma/yayın olarak kurulmuştu. GitHub API bu depoda yalnız CI workflow’unu gösterdi. Windows çalıştırıcısı `research/ct_forward_daily.py`, private yedek repo `ErayKulkizaga/CottonLensAI`. Çalışma makbuzları `ct-forward-records-2026-q4` altında. Başlangıç state’i `ct-forward-initial-state-daily-20261030`, ilk v1 denetimi `ct-first-publication-audit-20261031`; v2’nin ayrıca commit’e alınmış 4 Kasım makbuzu var.

**Kesinlik** — kesin.

**Kaynak** — 16.11.2026 `Get-ScheduledTask` / `Get-ScheduledTaskInfo`, GitHub Actions workflow API’si, `README.md:15–17`, `docs/archive/price-phase/CT_FORWARD_DAILY_20261030.md:3–7,32–71`, `docs/archive/price-phase/STATUS.md:178–202`. [Yayın release’i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-forward-records-2026-q4).

**Claude nereye bakmalı** — Bu makinenin Task Scheduler kaydı ile private release’in capture/publish asset’lerine. Geçmişte planlanan saatleri güncel çalışma kanıtı olarak kullanmamalı.

### 17. Origin sayısı ve ileri QLIKE

**Cevap** — İki gerçek zamanında yayın, iki ayrı kohort: v1’de 30 Ekim origin → 2 Kasım hedef (31 Ekim yayını), v2 first-observed’da 3 Kasım origin → 4 Kasım hedef (4 Kasım yayını). Geçerli v2 zinciri kendi başına **1** yayın. Release metadata’sında iki `publish` ZIP’i ve makbuzları var. V1, 2 Kasım origin’ini kaçırdı. 252/252 zamanında yayın şartı v1 için bozuldu, v2’de de görev kapalıyken tam örnek oluşmadı. İleri QLIKE farkı **bilmiyorum**: ara performans skoru üretilmemiş ve bu incelemede hesaplanmadı. Yaklaşık +0,0177 QLIKE, 2016–2023 tarihsel ikincil kolun sonucudur. V1’in takvim uzatmasında son hedef 2 Kasım 2027, yeni doğan kohort için başlangıç değiştiğinden ayrı takvim kilidi esas alınır. Eksik yayınlı dönem sonunda protokolün öngördüğü çıktı tanımlayıcıdır.

**Kesinlik** — kesin. Yayın sayıları ve skorun durumuna ilişkin kayıt belirli; ara skor sayısı yok.

**Kaynak** — `docs/archive/price-phase/END_TO_END_AUDIT.md:303–323`, `docs/archive/price-phase/STATUS.md:178–202,231–235`, `docs/archive/price-phase/CT_FORWARD_DAILY_20261030.md:78–96`, `research/evidence/ct-forward-first-observed-publication-audit-20261104.json:80–126`. [Yayın release’i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-forward-records-2026-q4).

**Claude nereye bakmalı** — İki `publish` asset’i, ayrı cohort lock’ları, dönem sonu `research/ct_forward_report.py` ve Q27’deki tarihsel belge.

### 18. Son başarılı yayın

**Cevap** — 4 Kasım 2026, Türkiye saati ile **03:20:05.128014**. GitHub publish ZIP’i yaklaşık 03:20:12’de tamamlandı. Task Scheduler son çalışma 4 Kasım 03:20:20, dönüş kodu 0; makbuz yayın ile scheduler zamanını ayrı ölçer. Bu, v2 origin 3 Kasım / target 4 Kasım kaydı.

**Kesinlik** — kesin.

**Kaynak** — `docs/archive/price-phase/END_TO_END_AUDIT.md:308–323`, `research/evidence/ct-forward-first-observed-publication-audit-20261104.json:84–120`, `CT-forward-20261104T002007Z-publish-e9e2669b.zip` metadata’sı ve yerel görev bilgisi. [Private release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-forward-records-2026-q4).

**Claude nereye bakmalı** — V2 makbuzunda kaynak, yayın ve GitHub yükleme zamanlarını ayrı alanlarda okuyabilir.

## F · Uygulama ve altyapı

### 19. Uygulama, dağıtım ve demo

**Cevap** — FastAPI fiyat/tahmin geçmişi, model karşılaştırması, açıklama ve senaryo çıkarımı API’si. Angular Dashboard, Model Lab, Replay, Sensitivity ve Why sayfaları. PostgreSQL + API + Nginx/Angular Docker Compose ile açılır. Rekonstrükte edilmiş geçmiş `backtest` ve zamanında kaydedilmiş `live` ayrıdır. Görüşme demosu için Docker Desktop Linux motoru açıkken depo kökünde `docker compose up --build`, ardından `http://localhost:8080` ve `http://localhost:8080/api/v1/openapi.json`. Development fixture banner’ı varsa örnek veri olarak anlatılmalı. Ölçülmüş eski demo için checksum’lı release ve import kimliği gerekir. Aktif learned fiyat üstünlüğü gösterilmedi, deneysel XGBoost sensitivity ana tahminden ayrı. Belgelenmiş yerel R2 sunum yolu da var. Ana checkout’ta `output/presentation/r2-demo/` altında artifact, `presentation.db` ve `presentation.json` mevcut. Doğru yerel ortam ve önceden build edilmiş frontend ile `python backend/scripts/presentation.py serve --name r2-demo --port 8080`, ardından `http://127.0.0.1:8080`. Bu seçenek tarihsel R2 verisini gösterir.

16 Kasım’da PR #77’nin `445d1fe` başında sekiz GitHub check kaydı başarılıydı (backend, frontend, ML/data ve Compose kontratı iki tetiklemede). GitHub Deployments API sıfır kayıt, Pages API 404 verdi. Harici bir backend/frontend deploy’u **bilmiyorum**. Bu inceleme sırasında Compose açılmadı. CI sonucu belirli commit ve test kapsamına aittir.

**Kesinlik** — kesin. Uygulama kapsamı, başlatma komutu ve CI doğrulandı. Harici deploy ve güncel uçtan uca çalıştırma açık.

**Kaynak** — `README.md:3–6,32–50`, `compose.yaml`, `.github/workflows/ci.yml`, `frontend/src/app/app.routes.ts`, `backend/app/api.py`, `docs/archive/INTERVIEW_DEMO.md:3–18`, `docs/archive/THREE_DAY_SPRINT.md:87–129`, [PR #77](https://github.com/ErayKulkizaga/CottonLensAI/pull/77) ve 16 Kasım GitHub API kontrolü.

**Claude nereye bakmalı** — Demo akışı için Interview Demo ve README. Gerçek demo materyali için ana checkout `runtime/artifacts/current/` manifesti, import kanıtı ve yerel release ZIP’leri. Deploy iddiası için çalışan endpoint veya harici platformun dağıtım kaydı gerekir.

### 20. ML paketi ve araştırma klasörü

**Cevap** — `ml/src/cottonlens_ml/` pip ile kullanılan veri/feature, fit, araştırma motoru, değerlendirme ve artifact üretim kodunu içerir (motor `research/` adlı alt pakette). Repo kökündeki `research/` deney sicili, receipted fit indeksi, ön kayıt/kanıt, arşiv/bağımsız doğrulama ve operasyona ait araştırma betiklerini içerir. Uygulamanın servis edilen production runtime katmanı `backend/app/` + `frontend/`, bunun girdileri doğrulanmış model paketleri ve DB. Eğitim ortamının kütüphaneleri backend image’da bulunmaz. Windows ileri test betikleri ayrı araştırma operasyonudur.

**Kesinlik** — kesin.

**Kaynak** — `docs/archive/phase2-notes/CLAUDE_HANDOFF.md:52–62,95–106`, `research/README.md`, `ml/pyproject.toml`, `backend/pyproject.toml`, `backend/app/artifact_importer.py`, `backend/app/runtime.py`.

**Claude nereye bakmalı** — Fit/eval için `ml/src/cottonlens_ml/research/engine.py`; arşiv için `research/storage.py`; çıkarım için backend runtime.

### 21. Colab GPU saatleri, GPU türleri, ödeme

**Cevap** — Toplam GPU saati ve ödenen tutar **bilmiyorum**. Eray, bilgisayarın bazen günlerce açık kaldığını hatırladığını bildirdi. Bu, kullanıcının beklediği takvim süresine ilişkin bir gözlemdir. Belgelenen `research-market-v1` compute toplamı 44.085 saniye (~12,25 saat). Bu fit-ledger compute süresidir; GPU aktif zamanı veya fatura birimiyle aynı ölçü değildir. 14 fitlik free-data pilot compute kaydı 24,1451 saniye, eski Round A için toplu tarihsel GPU saati eksik. Round A ve market-v1 yerel `ready.json → gpu.name` alanında Tesla T4; L4 checkpoint’leri ve Kasım L4 sentetik recovery çalışmaları doğrulanmış. `a100` adlı Kasım arşivlerinin bir kısmı L4 sonucu içeriyor; dosya adından A100 kullanımı çıkarılamaz. v1 raporu fiziksel GPU aygıtını gösteriyor, satır 10 GPU modelini belirtmiyor. Toplam maliyet için plan/bütçe konuşmaları gerçekleşen ödemeyi göstermiyor.

**Kesinlik** — bilmiyorum. Parçalı compute süreleri araç kayıtlarından geliyor.

**Kaynak** — `output/reports/research-market-v1-ablation-review-20261001.json` (`timing`), `docs/archive/RESEARCH_RESET_20260928.md:18–20`, `docs/archive/CONTINUOUS_RESEARCH.md:3–7`, `docs/archive/price-phase/DATA_GPU_PROGRESS_20261106.md:165,181,313`, v1 TXT:10, `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md:55–57`.

**Claude nereye bakmalı** — Colab oturum/cu kullanım geçmişi ve varsa ödeme kayıtları. Yerel Drive notebook çıktıları ve her ayrı run’ın `result.telemetry` alanı GPU modelini doğrulamak için daha uygun. Eksik oturumları fit sayısından saate dönüştürmemeli.

## H · Paket kontrolü

### 26. Satır bazında düzeltme listesi

**Cevap** — Aşağıda 03 ve 04’ün Phase 1 anlatısında düzeltilmesi veya kapsamı belirtilmesi gereken ifadeler var. 04’ün Phase 2 kısmı bu denetimin kapsamı dışında. 03’teki Link 1 köprüsünü kendi ön kaydı ve makbuzuyla karşılaştırdım.

#### `03_PHASE1_PRICE_AND_RISK.md`

| Satır | Düzeltme | Kaynak |
| --- | --- | --- |
| 3 | `codex_fit_recipes.csv` eldeki defter tariflerinin dökümü. Legacy aileler ve ablation’ın tamamı bu tabloda yer almıyor. `codex_experiments.csv` skor alanları boş sicil satırlarını da içeriyor. | `research/package/build_catalogs.py:42–86`; Q1–Q3, Q12, Q27. |
| 6–8 | CT=F günlük bar `Close` proxy’si için “settlement” genellemesi yapma. Gerçek named hedef ve fiyat alanı kaynak bazında yazılmalı. Drift ve çoğunluk yönü fiyat/referans kontrolleri; T+1 varyansın referansı EWMA. | `docs/archive/price-phase/PRICE_SEMANTICS_AUDIT_20261005.md`; `docs/archive/CONTINUOUS_RESEARCH.md:20–22`; `docs/archive/MODEL_EXPERIMENTS.md:41–49`. |
| 9–15 | Eşikler operasyonel/pratik kabul kuralları. Eski paketlerde ek LSTM–XGB karşılaştırması vardı. Risk kapısı belirli ön kayıtlara ait; canlı v4 için ayrıca 252/252 zamanında yayın şartı var. | `docs/archive/MODEL_EXPERIMENTS.md:41–49`; `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md`; `docs/archive/price-phase/CT_FORWARD_DAILY_20261030.md:78–96`. |
| 17–23 | Tasarımı çalışmaya göre yaz. V1/R2/sprint dört 126-origin blokta 504 origin kullanır. Round A 8×126, üç önceki 63’lük inner blok ve ayrı erken durdurma segmenti kullanır. 2.006 ve 21 origin cadence belirli full-year işlerindir. 2024+ seçimsiz audit sonraki kuraldır; v1 raporunda eski politika “rejection only” olarak yazılı. Round A bootstrap ana blok 20, duyarlılık 10/40; 20/60 daha sonraki çalışmalarda. Risk shrinkage EWMA ile varyans karışımıdır. | V1 TXT:29–37,73–102; `docs/archive/CONTINUOUS_RESEARCH.md:20–28`; `docs/archive/price-phase/CT_EWMA_HAR_CONTROL_20261014.md:24–30`. |
| 25–27 | “Her fit” ve “byte-equal” kapsamını daralt. Yerel checkpoint makbuzu ile Git’teki derlenmiş defter farklı. Eski TF-placement/On-call payload açığı var; CatBoost GPU taze refitte tolerans, bazı arşiv denetimleri saklı state’ten sıfır-fit replay kullanır. Açık smoke/reproduction kimlikleri ve yarım fit retry ayrı. | `research/README.md:672`; `docs/archive/phase2-notes/CLAUDE_HANDOFF.md:171–191`; `docs/archive/CONTINUOUS_RESEARCH.md:7–9,30–36`. |
| 33–38 | 3.905 label’lı origin veri evreni; tablo 504 dış origin/ufuk. Ridge de çalıştırıldı. Aynı v1’in metrikleri ve daha sonra geri çekilen legacy olumlu LSTM iddiasını ayrı kaydet. | V1 TXT:29–82; `research/data-gpu-defects.json:130–139`. |
| 40–42 | Dört LSTM adayı **fold başına**, artı final split için dört. Her fit iki çıktı üretir. Epoch 1, geniş validation split’indeki 32 hücre/0,20 adayına ait. Fold’lara genellenemez. | Q4–Q6. |
| 44–45 | V1 bulguları doğru. R2’de price-MAE early stopping XGBoost içindir; LSTM scaled-target validation MAE’si üzerinden epoch seçer. R2’nin farklı veri kimliği, ayrı v2 kayıp eğrisi ve exact v1 replay sınırını ekle. | `docs/archive/MODEL_EXPERIMENTS.md:18–38,49`; `docs/archive/DAY1_R2_EVIDENCE_REVIEW.md:3–23,38–45`. |
| 46 | Sprint için sıfır farkı seçilmiş reçetelerin benchmark reproduction çıktılarıyla sınırla. Aynı sonucun diğer erken deneylere taşındığı veya CUDA ile yapıldığı izlenimini verme. | Q2; sprint raporu; `docs/archive/RESEARCH_RESET_20260928.md:20–24`. |
| 48–52 | Round A sonucunun 28 Eylül tarihli reset’te kayıtlı olduğunu belirt; 30 Eylül–1 Ekim başlığı ikisini birden tarihlendirmemeli. 7.331, `cotton-research-v1` profilinin korunmuş defter toplamı. Dört aileye ait Round A sonuçları 64 fold kararı/64 dış vektörden gelir; 7.331’in tüm arama olduğunu söyleme. Ridge/ElasticNet CPU referansı. | `docs/archive/RESEARCH_RESET_20260928.md:1–18`; `output/reports/research-v2-A-saved-results.json`; Q9–Q11, Q27. |
| 53–55 | 7.320 XGB market fit’i 257 tarif satırına bağlı. `params` JSON’ları tekilleştirilince **179** set çıkıyor. Bu kesitte hedefler 119 scaled_log, 134 price_delta ve **4 raw_log** tarif satırı. 182 ve yalnız iki hedef ifadesini düzelt. | `research/trials.json`; Q10, Q27. |
| 59–65 | Performans tablosu 1.008 origin/ufuk, görülmüş tarih. “All gates failed” yerine hiçbir aday birleşik kapıyı geçmedi. Örneğin XGB T+5 %55,26 ile yön alt kapısını geçti. | `docs/archive/RESEARCH_RESET_20260928.md:5–16`. |
| 66–72 | B/C bütçeleri ufuk başına, C 48 aile **ve** ufuk başına. “Only one … each … CatBoost” hatalı: defterde iki CatBoost kontrolü. MLP sekans penceresi 1, LSTM/TCN 20. TCN kontrolünün eski genel `passed` etiketi hatalı: bilinen-sinyal kazancı −%23,593133, aile bazında öğrenme başarısı doğrulanmadı. Yedeklerde sentetik kontrol sayısı defter kesitinden geniş olabilir. | `docs/archive/CONTINUOUS_RESEARCH.md:11–18`; `research/trials.json`; Q8–Q9. |
| 73–76 | 20.360 ayrı deneyin makbuz toplamı: 20.352 piyasa fiti ve 8 diagnostik; 1.008 origin **ufuk başına**. Compute ~12,25 saat; GPU aktif/billed süre değil. `cotton_no_volume` DXY/WTI ve çapraz korelasyonları da çıkarır. Genişletilmiş availability kaynak ilk-yayın/vintage kabulü değildir. Rapor doğrulaması tüm payload’ları kapsamadı. | `output/reports/research-market-v1-ablation-review-20261001.json`; `ml/src/cottonlens_ml/research/ablation.py:18–32`; Q12. |
| 77 | 6.648 yalnız eski XGBoost T+1 lineage. Hedef, loss ve diğer hiperparametreler birlikte değiştiğinden farklı yıllardaki seçim nedensel loss etkisi ölçmüyor; eşlenmiş ayrı loss kontrolü var. | `docs/archive/price-phase/LEGACY_LOSS_LINEAGE_RESULT_20261010.md:3–28,35–53`; `docs/archive/price-phase/PAIRED_PRICE_LOSS_RESULT_20261010.md`. |
| 79–98 | 1.855 altı standart profilin sayısı. WASDE 1.162, AMS 231, on-call 222 defter kaydı. On-call −%1,32 bir **kısmi**, 502-origin `oncall_L2` sicil satırı; tamamlanmış 2.006-origin replay’de kaynak L1/L2/L6 −%0,272/−%0,026/−%0,385. AMS/WASDE kapsamı farklı. L1/L2/L6 geçmiş saat/vintage varsayımları evrensel doğrulanmış ilk yayın saati değil. 84’te seçilmiş yön görünmüyor; 98’de sıfıra shrinkage genellemesi yerine aktif oran yaz (on-call kısmi %100). | `docs/archive/MASTER_PLAN_20261001.md:562–564`; `research/registry.json`; paket `data/codex_experiments.csv:64`. |
| 101–104 | “Timing loss is not the main mechanism” yalnız eşlenmiş sabit DXY/WTI Ridge kolunun bulgusu. “No gain”/“Noise-level” yerine ilgili kıyas ve belirsizlik. +%0,22 ölçütü `registry.json` içindeki selected fiyat satırlarına ait maksimum; diğer çalışmaların tamamına uygulanmaz. | `docs/archive/price-phase/AVAILABILITY_CLOCK_RESULT_20261005.md`; `docs/archive/MASTER_PLAN_20261001.md:573–583`; Q27. |
| 105–109 | Text/regional sonuçları sabit pilot/ortak kapsam olarak ver. Texas numeric D0/D1 için −%0,02744 / −%0,25559 ve raw D0 −%3,75575; −%0,03 bütün kolların sonucu değil. Zayıf sinyal öğretmeni belirli senaryolar için, “59/80 değişti” injected set içindir. | `docs/archive/price-phase/NASS_REGIONAL_T1_RESULT_20261009.md`; `docs/archive/price-phase/WEAK_SIGNAL_RESULT_20261009.md`; `docs/archive/price-phase/SELECTION_STABILITY_RESULT_20261010.md`. |
| 110–113 | Paired loss’taki +%1,46 karşılaştırması MSE koluna karşı; Naive sonucu ayrıca gösterilmeli. Kalibrasyonda “unchanged” karar ve pratik ölçek için. T+1 curve tamamı weight 0 olan programın kapsadığı origin’lere aittir. | `docs/archive/price-phase/PAIRED_PRICE_LOSS_RESULT_20261010.md`; `docs/archive/price-phase/PRICE_MAE_CALIBRATION_RESULT_20261010.md`; `docs/archive/price-phase/CONTRACT_CURVE_RESULT_20261010.md`. |
| 115–118 | Roll’lu proxy’de fiyat değişimi, aynı kontratı tutmanın getirisi olarak yorumlanamaz. Named label/capacity sonucu dar 2023 kohortunu kapsar. Öğretmen, sabit tarifin üretilen etkiyi öğrenebildiğini gösterir; piyasadaki tüm başarısızlıkların eğitim kusurundan bağımsız olduğunu göstermez. | `docs/archive/price-phase/NAMED_CAPACITY_CONTROL_T5_20261010.md`; `docs/archive/price-phase/NAMED_WEAK_SIGNAL_EXECUTION_20261012.md`; `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md:44–54`. |
| 121–129 | Risk target günlük close-to-close karesel getiri. HAR parametrelerini kol bazında ver (own-level/DXY beş parametre). CT raw +0,01747463 ile seçili +0,00136570 farklı. “One historical risk candidate” ifadesini tarihsel kapıları geçen önceden kayıtlı **ikincil** own-level kol olarak nitele, DXY birincil sonuç inconclusive. | `docs/archive/price-phase/CT_EWMA_HAR_CONTROL_20261014.md:7–12`; `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md:95–108`. |
| 130 | 95,9% yoğunlaşmayı aktarmaya ek olarak üst %1 çıkarıldığında katkı +0,00073073’e düşüyor. 169 katsayı bulgusu extra `log(EWMA)`’nın bu modeldeki işaretine ait. | `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md:97–100,122–141`. |
| 131–136 | “Daily automated publishing” tarihçe. V1 30 Ekim’de başladı, bir yayın ve 2 Kasım missing. V2 başlangıç 3 Kasım, bir yayın; scheduler kapalı. 252 slot / 31 fit tasarım bütçesi ve ilk takvimin hedefi, tamamlanmış ileri test veya skorlama değil. +0,00009 recent audit skoru, pre-2010 seçili EWMA’ya eşit; ikisi de ayrı tarihsel test. | Q16–Q18; `docs/archive/price-phase/STATUS.md:52–76,93–105,178–202,231–235`. |
| 139–142 | 217 evaluation (ve 218 history) tarih, geçmiş teslim ayı kesin tanığı. 3.320 **kimlik** hücresi ayrı overlay’de; 110 HAR girdisi 863 doğrulanmış pencereden. Kapsamı ilgili yıllık AMS named sonuçlarıyla sınırlı tut. Yahoo değişiklik satırına v1’in reddedilen origin’ini ve v2’nin ayrı kilidini ekle. ICE 36/75 sayısı kaynak doğrulamasının kapsadığı witness set. | `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md:44–49`; `docs/archive/price-phase/AMS_CONTRACT_RECONCILIATION_20261029.md:12–39,58–81`; Q13–Q14. |
| 145–149 | 5 feature, 2.006 origin ve 338+4 sayısı iki kol toplamı; birincil Quantile için 169+2. STOP sabit reçete ve araştırma önceliği kararı. | `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md`; Q9. |
| 151–157 | 15 Kasım Link 1 Claude Phase 2 köprüsü; 7 Kasım’a kadarki Codex deneyleriyle aynı dönem değil. Ön kayıt all-cotton der, gerçekleşen seri Upland; iki eksik previous-forecast olayı ve oracle n=32 belirtilmeli. H2 p=0,07549245 (üç ondalıkta 0,075). H1 eşlenmiş ay permütasyonundaki ilişki; “reacts” nedensellik kuruyor. H2/combined gate başarısız, üç büyük gün çıkarılınca H1 de başarısız. “No free archive exists” için bu kayıtlarda evrensel ispat yok; konsensüs serisi çalışmaya dahil edilmedi. | `docs/prereg/LINK1_EVENT_STUDY_PREREG_20261115.md:11–18,28–53`; `research/evidence/link1-event-study-20261115.json:1–66`. |
| 160–163 | Tarihsel denenen protokollerde pratik fiyat kapısı geçilmedi. “Returns are close to a random walk for every feature set” istatistiksel random-walk tespiti gibi okunur. 113 skor sicil hücreleri, bağımsız 113 deney veya bütün çalışmaların maksimumu değil. 4.597 sentetik fit pozitif ve negatif kontrolleri içerir; pipeline’a genel beraat vermez. Named kusurun kapsamını belirt; forward görevi durdu. | Q16–Q18, Q27; `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md:42–60`. |

#### `04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md`

| Satır | Düzeltme | Kaynak |
| --- | --- | --- |
| 3,7–11 | 14.723, defterdeki XGBoost toplamı. Ayrı legacy sprint/v1/R2 ve market-v1 dahil değil. `7.320 market +6 synthetic` Round A profilinin XGB alt kümesi; bütün Round A eğitim sayısı değil. | Q1–Q3, Q9, Q12, Q27. |
| 28–38 | 387 sayısı defterde XGB aileli tarif satırı; parametre aralıkları bunların betimlemesi. Belirli parametrelerin kaç tarifte mevcut olduğu CSV’nin `n_recipes` sütununda. Late-stage tariflerin search boundary ile aynı kapsamı yok. | `research/trials.json`; `research/package/build_catalogs.py:42–71`. |
| 40 | Dağılım sadece log ölçek değil. Deterministik NumPy örnekleme, uniform/ayrık draw’lar ve `target`/`loss` döngüsü. Her fold/ufuk/aile başına 5 finalist, seed 17/42/101. | Q10–Q11; `ml/src/cottonlens_ml/research/search.py:6–40`, `engine.py:283–299`. |
| 41 | Sabit depth-2 genellemesini standardize keşif kollarıyla sınırla. AMS ve on-call ayrı protokol/kapsam; her sonraki pilot aynı reçete değil. | `research/trials.json`; `docs/archive/MASTER_PLAN_20261001.md:267–275,562–564`. |
| 42 | 387 XGB tarif satırında objective **223** squared, **164** absolute. Target **201** scaled_log, **182** price_delta, **4** raw_log. Paket sayıları farklı paydaları birleştiriyor. | `research/trials.json` XGB recipe alanlarının yeniden sayımı. |
| 43–44 | “CPU elsewhere” hatalı; market-v1 GPU deniyor, bazı keşif makbuzlarında L4/CUDA doğrulandı. Seed 42 ana arama ve sabit pilotların seed’i, Round A finalistleri 17/42/101. Price-MAE early stopping protokol ve eğitilen kol bazında ifade edilmeli. Native `pred_contribs` açıklaması deneysel XGB modeliyle ilişkilendirilmeli. | `docs/archive/CONTINUOUS_RESEARCH.md:11–17`; ana checkout `docs/MASTER_PLAN_20261001.md:707,800`; `c33d6ae:ml/src/cottonlens_ml/export.py`; Q10. |
| 52–56 | Ridge **13.901**, Quantile **171**. İki kolun 342 fit’inden Quantile 171, Ridge 171. Development/market ve synthetic role kayıtları ayrılmalı. 129, Ridge tarif satırı. 21 cadence ve shrinkage her Ridge tarifinde ortak değil. +%0,22 en iyi selected registry değeridir. | `research/trials.json`; `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md`; Q9/Q27. |
| 58–72 | HAR 3.048 doğru. `own_har_k − log(EWMA)` ifadesini log-feature koluna ve ilgili offset’e bağla. EWMA’ya geri dönüş ilgili log-link dahil tüm lineer düzeltme sıfır olduğunda veya seçilen karışım ağırlığı 0 iken olur. +0,0177 tarihsel ikincil sonuç; görev kapalı. | `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md:95–108`; `docs/archive/price-phase/CT_EWMA_HAR_CONTROL_20261014.md:7–30`; Q16–Q18. |
| 74–78 | GARCH uygulama ayarları doğru, sonuç full-year tarihsel protokole ait. 356 tamamlanmış ElasticNet defter fit’inin parametreleri doğru; Round A ElasticNet arama uzayı ayrıdır (geniş α ve l1_ratio). | `ml/src/cottonlens_ml/research/volatility.py:133`, `search.py:28–31`; `research/trials.json`; `docs/archive/price-phase/RISK_EVIDENCE_AUDIT_20261012.md`. |
| 80–85 | 342 toplam; Quantile 169 piyasa+2 sentetik, Ridge 169+2. Ham fiyat-delta, `highs`, thread 1, 120 s. EWMA λ=0,94, burada anlatılan ana risk protokollerinin sabit referansı; başka tarifi/araştırma önerisini “everywhere” ile kapsamamalı. | `research/trials.json`; `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md`; Q9. |
| 87–90 | İki CatBoost fit **defter** sayısı. Daha geniş arşivde gerçek piyasa fit’leri var. Aday sayısına ek olarak market-v1’de **10.176 CatBoost piyasa fiti** ve üç CatBoost tanı kontrolü sayıldı. Taze reproduction için tolerans kuralı var, bu aramaların bütün CatBoost modelleri için o kontrolün geçtiği kanıtlanmadı. | `docs/archive/CONTINUOUS_RESEARCH.md:30–36`; Q9, Q11–Q12. |
| 92–99 | 23 Eylül V1: 4 aday **fold başına**, iki horizon çıkışı ortak. Epoch 1 yalnız final validation split’i. DG-008 farklı bir artifact’e, 22 Eylül `v20260922-1450` çalışmasına ait. O dönemde 6 aday vardı. GradientExplainer, seçilen LSTM’i export edebilmek için yazılan koşullu yol; v1 ZIP’te seçilen tahmin Naive ve LSTM payload’u yok. DG-008’in özgün vektörü doğrulanamadı, yerel checkpoint ve raporlar bulundu. Round C bütçesi aile ve ufuk başına; koşullu durdurma kararını ekle. | Q4–Q8; `c33d6ae:ml/src/cottonlens_ml/export.py:75–107,280–292`; v1 ZIP üye listesi. |

**Kesinlik** — kesin. Düzeltmeler gösterilen dosya sürümü ve kanıt kapsamına bağlı.

**Kaynak** — Satırlar Claude worktree’deki `03_PHASE1_PRICE_AND_RISK.md` / `04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md`, commit `1247455` (bu iki dosya için `59177d7` ile aynı içerik). Üçüncü sütun ilgili ölçüm ve kodu gösterir.

**Claude nereye bakmalı** — Her tablo satırının üçüncü sütunundaki özgün kayıt ve ilgili cevap. Özellikle Q27 kapsamlarını paket özeti ve görüşme cevaplarına da yansıtabilir.

### 27. Ana sayıların doğrulanması

**Cevap** —

| İddia | Yeniden sayım / doğrulama |
| --- | --- |
| 32.934 toplam | **Doğru.** `trials.json` 569 toplulaştırılmış kayıt, toplam 32.934 `completed_fits`. 28.245 `market_fit_receipt` + `synthetic*` 4.597 + 61 `historical_walk_forward_audit` + 31 `archival_replication_same_contract_risk` (92). Kapsam eldeki defter. |
| 14.723 XGBoost | **Doğru.** Aile bazında doğrudan yeniden sayıldı. `cotton-research-v1` 7.320 market +6 synthetic. Dokuz kaynak keşfi 6.729 (FX 1.020, CFTC 1.008, weather 996, FAS 996, NASS 984, crop 972, WASDE 528, on-call 141, AMS 84). Diğer profiller 668. Yaklaşık 6.500 yerine 6.729 yaz. |
| Registry max +%0,22, medyan −%0,65 | **Doğru, doğru payda ile.** 239 sicil satırında `metrics.selected.gain_pct` dolu 113 satır. Medyan −%0,6481903602, maksimum +%0,2171214239 (`fundamental-joint-pilot-v1/joint_L1`, T+5, 2.006 origin, raw −%6,6417598771). Aynı dönem/özellik deneyleri bağımlı olabilir. |
| market-v1 en iyi +%0,763 | **Doğru.** `ablate-expanded_availability` XGB T+5 +%0,7631, yön %57,54, 5/8, gain CI yaklaşık [−%1,4156, +%2,1838], birleşik kapı geçmedi. 16 hücrenin en yükseği. |
| Own-level +0,0177 QLIKE | **Doğru.** Seçilmiş tarihsel, ön kayıtlı ikincil own-level kol. +0,01770104; CI20 [+0,00428236, +0,02989275], CI60 [+0,00470139, +0,02957616], 7/8, trim +0,00073073. İleri kohort farklı. |
| Link 1 1,42; p=0,004 | **Doğru yuvarlama.** 38 olay / 1.164 uygun karşılaştırma günü, H1 oranı 1,4187095749 ve p=0,0037996200. Üç büyük gün hariç 1,1508007922 ve p=0,1565843416. H2 ve birleşik kapı başarısız. Çalışma Upland serisine ait, all-cotton ön kaydından sapma makbuzda yazılı. |

**Kesinlik** — kesin. İlk üçü yapılandırılmış dosyalardan yeniden hesapladım. Ablation, HAR ve Link 1 değerlerini özgün rapor/makbuzla karşılaştırdım. Ham piyasa verisinden yeni fit, yeni bootstrap veya mühürlü Link 1 testi yapılmadı.

**Kaynak** — `research/trials.json` (`records[].completed_fits`, `kind`, `profile`, `recipes[].family`); `research/registry.json` (`records[].metrics.selected.gain_pct`), örnek satır paket `data/codex_experiments.csv:5`; `output/reports/research-market-v1-ablation-review-20261001.json` (`results`); `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md:95–101`; Claude `research/evidence/link1-event-study-20261115.json:1–66`.

**Claude nereye bakmalı** — Katalogdan önce ham JSON ve özgün makbuz. Defterin toplamı için `completed_fits`, dağılım için `kind`, XGBoost için doğrudan aile filtresi. Registry max/medyan yalnız dolu selected gain alanları. Phase 1 ve Link 1’i farklı zaman ve protokol başlıklarında tutabilir.

## A · Defterden önceki çalışmalar, 21–25 Eylül

### 1. V1’de kaç model eğitildi?

**Cevap** — `v20260923-0652` raporunda **80 XGBoost tuning fiti** (8 aday × 2 ufuk × 5 bölüm) ve **20 LSTM tuning fiti** (4 aday × 5 bölüm, iki çıkış ortak). Metin 80 XGBoost ve 40 LSTM trial satırında `resumed=False` gösteriyor. LSTM satırları iki ufukta tekrarlandığından 40 satır 20 eğitim. Beş bölüm, dört outer fold ve final train/validation çalışması. Kod ve tamamlanan rapor ayrıca ablation için **24 sabit XGBoost fiti** (3 feature set × 2 ufuk × 4 fold) ve **10 Ridge fiti** (2 ufuk × 5 bölüm) gösteriyor. Böylece bu raporun deney akışında **134 fit**, aile olarak **104 XGBoost, 20 LSTM, 10 Ridge**. Naive bir fit gerektirmez. Smoke/probe veya daha eski başarısız oturumlar bu sayıma dahil değil; 21–23 Eylül’deki tüm Colab donanım işi toplamı **bilmiyorum**.

**Kesinlik** — kesin. 134, tamamlanmış özgün v1 raporunun ölçülebilir deney akışına ait.

**Kaynak** — `output/claude-audit/drive/reports/cottonlens-results-v20260923-0652.txt:84–93,109–255` ve aynı adlı JSON. `c33d6ae:ml/src/cottonlens_ml/pipeline.py:28–38`, `walkforward.py:125–196`, `training.py:79–209,233–349` (aynı paket). Özgün release ZIP dış SHA-256 rapor ve `.zip.sha256` ile uyumlu, CRC geçti.

**Claude nereye bakmalı** — Geri yükleme altındaki `reports/`, `checkpoints/walkforward-fold-*/` ve `artifacts/releases/`. Başka erken smoke/başarısız denemeler için özgün Colab notebook execution log’u veya MLflow SQLite gerekir.

### 2. Düzeltilmiş v2 ve tabular sprint

**Cevap** — R2 `v20260924-185501-519b9c1a` için readiness `e153054a…` dizininde **97 completed native checkpoint makbuzu** var: **76 XGBoost** (64 tuning, 8 fold refit, 2 audit, 2 deployment) ve **21 LSTM** (16 tuning, 4 fold refit, 1 audit). 97 makbuzun başvurduğu checkpoint dosyalarının SHA-256 değerleri eşleşti. Makbuz sayısı Ridge ve bağımsız smoke’ları kapsamaz; R2’nin tüm fitleri için tek eksiksiz sayaç **bilmiyorum**. R2’de iki ufukta da Naive seçildi.

`v20260925-074708-5d7bdf26` sprint arşivi **204 XGBoost checkpoint completion** içeriyor: development 168, benchmark 16, reproduction 16, evaluation_backtest 2 ve deployment_live 2. 204 makbuzun işaret ettiği checkpoint dosyalarının SHA-256 değerleri eşleşti. Seçilen benchmark reçetelerinin reproduction farkı sıfır. Native checkpoint toplamı geçmiş deneme/CPU probing süresi sayısı değildir.

İki çalışma da Eylül’de ayrı checkpoint/report formatında yapılmıştı. 5 Ekim’de eklenen Git fit defteri daha sonraki research ledger makbuzlarından oluşturuldu. Legacy pipeline makbuzlarının neden ayrıca dönüştürülmediğine dair ayrı karar kaydı **bilmiyorum**.

**Kesinlik** — kesin. Yerel makbuz kapsamları sayıldı, Eylül oturumlarının eksiksiz toplamı açık.

**Kaynak** — `output/claude-audit/drive/checkpoints/e153054a49965a3e176aedc4105280232dcda62ac81cfac5c9889cd31bb92b82/` ve R2 raporu/ZIP’i; `docs/archive/DAY1_R2_EVIDENCE_REVIEW.md:3–23,38–45`; `CottonLensAI Colab/CottonLensAI/experiments/tabular-sprint-v1/checkpoints/`, `reports/cottonlens-results-v20260925-074708-5d7bdf26.{json,txt}`; commit `3d5f7f0`.

**Claude nereye bakmalı** — R2 readiness dizininde `*.complete.json`; sprint için `development`, `benchmark`, `reproduction`, `evaluation_backtest`, `deployment_live` alt dizinleri. Her ikisi de kendi tarihli release ZIP ve dış checksum’u ile doğrulanabilir. Bu adlar Colab artifact release sürümleri; private GitHub git tag’i ile aynı şey değil.

### 3. Defterin başlama tarihi

**Cevap** — Repoda `research/trials.json` ilk kez **5 Ekim 2026 11:09:40 TR**, `3d5f7f0` (“Curate research history, preserve evidence and add experiment lookup”) ile tutulmaya başladı. Deney başına checkpoint/ledger bundan önce vardı. `2003f72` (1 Ekim) motoru Git’e getirir; 28 Eylül reset’i daha eski `research-v2-tf-placement` / free-data ledger’larını zaten tarif eder. Eylül v1/R2/tabular dışında `research-v1` diagnostikleri, free-data pilot ve market-v1 ablation gibi kayıtları ayrıca ara. “Hiç kayda girmemiş başka bir fit” için tamamlanmış, eksiksiz oturum envanteri **bilmiyorum**.

**Kesinlik** — kesin. Git defterinin ilk commit’i ve daha erken ayrı kayıtlar doğrulandı.

**Kaynak** — `git log --all --diff-filter=A -- research/trials.json`; `3d5f7f0`, `2003f72`; `docs/archive/CONTINUOUS_RESEARCH.md:24–26`; `docs/archive/RESEARCH_RESET_20260928.md:1–20`; Colab `experiments/` arşivi.

**Claude nereye bakmalı** — Git için `git show 3d5f7f0:research/trials.json`; bundan önceki deney için deney dizininin kendi zaman damgaları, `ledger/completed`, report/MLflow. Git’e eklenme tarihi fit tarihi değildir.

## C · Diğer model aileleri

### 9. CatBoost ve Quantile

**Cevap** — **CatBoost defteri 2 fit**. İkisi de `cotton-research-v1` sentetik known-signal (depth 6, lr 0,1, l2_leaf_reg 1, seed 42, window 1, scaled_log, squarederror tarif etiketi). Gerçek piyasa CatBoost aramaları `research-v2-tf-placement` Round A ve `research-market-v1` ablation’da kayıtlı. Round A 128 GPU aday/ufuk, market-v1 16 aday/kol/ufuk/fold. Arama depth 4–8 ayrık uniform, lr [0,01, 0,15] log-uniform, l2 [1, 100] log-uniform. Model yolunda GPU zorunlu, early stopping için fiyat MAE metriği ve her adayın hedef/loss tarifi kullanılır. Round A CatBoost’un tüm oturumlarını kapsayan kesin toplam fit sayısı **bilmiyorum**. Yerel market-v1 arşivinde 10.176 CatBoost piyasa fiti ve üç CatBoost tanı kontrolü var. Round A yerel `ledger/completed` arşivi kısmi (4.107 tarifli makbuz; 3.312 ElasticNet, 792 Ridge, 2 CatBoost, 1 XGB). Bu kesitten bütün çalışmanın sayısı çıkarılamaz.

**Quantile defteri 171 fit**: `price-only-median-t1-v1` 169 piyasa +2 sentetik. Ridge karşı kolu da 169+2; 342 iki kol toplamı. QuantileRegressor, q=0,5, α=0,01, `fit_intercept=True`, `highs`, 1 thread, 120 saniye zaman sınırı; T+1 ham price_delta, 5 fiyat feature’ı, seed 42 tarif kimliğinde, cadence 21. `hyperparameter_ranges.csv:2–4,15–16` kendi sayısal kapsamı için doğru: CatBoost 2 ve Quantile 3 **tarif satırı** (169 toplu piyasa, iki tekil sentetik). Fit sayısı, solver ve boolean seçenekler, piyasa CatBoost arama uzayı bu CSV’den okunmaz.

**Kesinlik** — kesin. Defter/ablasyon kapsamları ile arama ayarları doğrulandı; Round A'nın geniş CatBoost toplamı açık.

**Kaynak** — `research/trials.json` CatBoost makbuzları `220bf782fab1…`, `8b110175f9f5…` ve `price-only-median-t1-v1`; `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md:63–83`; `ml/src/cottonlens_ml/research/search.py:6–40`, `models.py:273–291`; paket `research/package/build_catalogs.py:42–71`. Ablation için Q12.

**Claude nereye bakmalı** — Colab `experiments/research-v2-tf-placement/decisions/A` ve `experiments/research-market-v1/ledger/completed`. Park edilmiş eski model payload’larını makbuz veya doğrulanmış restore sayısıyla karıştırmamalı.

### 10. Round A 128 adayın örneklenmesi

**Cevap** — Kod `numpy.random.default_rng` kullanır; RNG seed’i aile, ufuk, task içeren `content_id`’nin ilk sekiz hex karakteri. Deterministik pseudo-random arama. XGB depth 1–8 ayrık uniform, eta 0,005–0,15, child weight 1–100 ve lambda 0,001–100 log-uniform. Alpha her üçüncü adayda sıfır, diğerlerinde 1e−6–10 log-uniform. `subsample` ve `colsample_bytree` 0,6–1 uniform. Hedef `scaled_log`/`price_delta` ve loss squared/absolute indeksle döndürülür. Model seed 42 ilk taramada farklı bir ayar. CatBoost aralığı Q9’da. Ridge/ElasticNet için α log-uniform 1e−5–100, ElasticNet l1_ratio uniform 0,05–0,95. Uzantılarda offset var. İç skor fiyat MAE / Naive oranıdır ve outer fold seçimde kullanılmaz.

**Kesinlik** — kesin.

**Kaynak** — `ml/src/cottonlens_ml/research/search.py:6–48`; özgün XGB T+1 draw ve onay yeniden kurma kodu `ml/src/cottonlens_ml/research/legacy_loss_lineage.py:40–79,162–203`.

**Claude nereye bakmalı** — `search.py`, sürüm eşitliği için `legacy-loss-lineage-20261010` frozen source ve yerel `decisions/A` tarif listeleri.

### 11. 5 aday × 3 seed sonuçları

**Cevap** — Her aile/ufuk/fold’da seed 42 iç skoruyla ilk beş. 17 ve 101’de ek fit, seed 42 ilk taramadan yeniden kullanılır. Üç seed’in iç skor ortalaması ve karmaşıklıkla kazanan seçilir; dış tahmin seçilen üç seed’in ortalamasıdır. 64 yerel karar JSON’u var. Örneğin `decisions/A/xgboost-t1-fold1.json`: tüm ilk-tarama satırları `trials`, seçilen adayın üç sonucu `chosen.seeds`. Karar formatı diğer dört finalistin üç-seed özetini saklamıyor. Dört aile, iki ufuk ve tüm fold’lar için ayrı, eksiksiz bir ilk beş tablosu **bilmiyorum**. Eski XGB T+1 onayı 6.648 arşiv bağlı makbuzdan sıfır yeni fit ile yeniden kuruldu; dosyadaki sekiz satır her fold’un seçilen adayını gösterir.

**Kesinlik** — kesin. Saklama ve yeniden kurma yerleri belirli, hazır bütün-finalist tablosu doğrulanmadı.

**Kaynak** — `ml/src/cottonlens_ml/research/engine.py:272–303,328–340`, `search.py:6`, `docs/archive/price-phase/LEGACY_LOSS_LINEAGE_RESULT_20261010.md:3–28`. `output/reports/research-v2-A-saved-results.json` dış sonuç özetidir. [Lineage release’i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/legacy-loss-lineage-20261010).

**Claude nereye bakmalı** — `CottonLensAI Colab/CottonLensAI/experiments/research-v2-tf-placement/decisions/A/`, eşleşen `ledger/completed/` ve lineage arşivi. Kaybolan payload’ların kapsamı için `research/README.md:672`.

## D · Ablasyon ve veri kaynakları

### 12. 20.360 alt-fit ve 16 sonuç hücresi

**Cevap** — `research-market-v1` kendi tamamlanmış fit makbuzlarını tutmuş. Yerel Colab arşivinde `ledger/completed/` ve `ledger/receipts/` altında **20.360’ar** JSON dosyası sayıldı; bunlar aynı kayıtların iki görünümü. Completed kayıtlarında 10.176 XGBoost + 10.176 CatBoost piyasa fiti; `diagnosis.json` ile eşlenen 8 kontrol (5 XGBoost, 3 CatBoost) var. Yerel karşılaştırma raporunun SHA-256 değeri özgün review JSON’uyla eşleşti. `research/trials.json` içinde bu deney profili/kimliği bulunmuyor. Proje teslimatı, report/outer çalışma birimlerini doğrulamış ve tüm model payload’ları için `all_model_payloads_verified: false` yazmış. Git indeksi seçili doğrulanmış receipt seti. Dışarıda bırakmanın açık insan kararı/gerekçesi **bilmiyorum**. 20.360 alt-fit ile defter toplamını tek bir doğrulanmış global sayım gibi kullanmamak gerekir.

Dört feature kolunun her biri XGBoost T+1/T+5 ve CatBoost T+1/T+5 ile çalıştı. Böylece 16 hücre × 8 dış yıl = 128 outer unit, ufuk başına 1.008 origin. Her unit 16 aday, arama/iç doğrulama/confirmation/dış refit alt fitleri içerir. Dört grubun özellikleri:

| Kol | Feature | Girdi farkı |
| --- | ---: | --- |
| `base` | 24 | Q4’teki pamuk/çapraz varlık/takvim tabanı. |
| `cotton_no_volume` | 14 | Tabanın iki pamuk hacim alanı, altı DXY/WTI getiri alanı ve iki pamuk–çapraz korelasyonu çıkar. |
| `expanded` | 39 | Taban + nedensel OHLCV/lag, rolling/gap/seri davranışı türevleri. Açık `days_since_previous_observation` hariç. |
| `expanded_availability` | 64 | 39 alan + önceki gözlemden gün farkı + 24 `age_<base>` gözlem aralığı. Train-fitted missingness maskeleri tüm gruplarda. |

Ablation giriş kapısı eşlenmiş iç skorlar üzerinden en az %0,5 iyileşme ve en az 5/8 iç dönem. Hiçbir grup iç admission’ı geçmedi. En iyi dış hücre +%0,7631 aynı bilgi üzerindeki küçük historical sonuçtur (Q27).

**Kesinlik** — kesin. Defter farkı, ayrı fit makbuzları ve feature grupları doğrulandı; insanın dışarıda bırakma nedeni kayıtta bulunamadı.

**Kaynak** — `output/reports/research-market-v1-ablation-review-20261001.json`, `docs/CLAUDE_PROJECT_HANDOFF_20261001.md:44–69,295` (ana checkout), `ml/src/cottonlens_ml/research/ablation.py:18–84`, arşivde `experiments/research-market-v1/{ready.json,ablation-plan.json,ablation-summary.json,ledger/}`. Kullanışlı karşılaştırma tablosu paket `data/codex_market_v1_ablation.csv:2–17`.

**Claude nereye bakmalı** — `CottonLensAI Colab/CottonLensAI/experiments/research-market-v1/`. `ready.json → identity.groups` tam feature listesi. `reports/comparison-9b2dadda57ec990d.json` ve fit makbuzları ayrı; arşiv payload tamlığını her kullanımda ayrıca kontrol et.

### 13. Yahoo first-observed

**Cevap** — 3 Kasım 03:05’te Yahoo, 30 Ekim CT=F kapanışını 78,13 yerine 77,83 verdi. V1 örtüşme kontrolü girdiyi reddetti ve 2 Kasım origin’i kaçtı. V2, her gün için **ilk kabul edilen** değeri taşır. Daha sonraki overlap’taki revizyonlar ve ham cevap ayrı makbuz; geçmiş state’te ilk kabul edilen fiyat kalır. Yeni tamamlanmış seanslar kendi gerçek kaynak gözetim zamanı/kesimi ve resmi takvim ile bağlanır. Tam seans, tarih ve overlap kontrolleri devam eder. Yeni V2 lock 3 Kasım origin’iyle, aynı model tarifi ve yeni kohorta özgü 10 ilk fit ile oluşturuldu; eski tahmin zinciri ayrı duruyor. Tarihsel Yahoo close’larının borsa final settlement/first print sertifikası bu operasyonla elde edilmez.

**Kesinlik** — kesin.

**Kaynak** — `docs/archive/price-phase/STATUS.md:178–202`; `research/ct_manual_forward.py:350–406,431–443`; `research/evidence/ct-forward-first-observed-publication-audit-20261104.json`; yerel `output/own-level-forward-v2-first-observed/cohort-lock.json`. [Private release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-forward-records-2026-q4).

**Claude nereye bakmalı** — V2 `inputs/bootstrap-revisions.json`, gözlem bazlı `overlap-revisions.json`, capture ham makbuzları ve ayrı v1/v2 release zincirleri.

### 14. AMS kontrat kimliği

**Cevap** — Yıllık sheet’te bazı teslim ayları gözlem tarihinden geçmiş görünüyordu. Örnek 26 Şubat 2020’de 65,47, günlük MP_CN001 `FUTURES TODAY` tablosunda Mayıs 2020; aynı fiyat yıllık dosyada Temmuz 2019. Yıllık fiyat vektörü ile tarihli günlük metindeki 10 sıralı fiyat/kontrat başlığı eşleştirildi. 1.262 ham günlük rapor doğrulandı, 332 gün × 10 kimlik = **3.320** değişiklik. 1.255 günde on fiyat eşleşti. 863 eksiksiz HAR penceresinden 110’unda girdi farklı. İlk 217 sayısı geçmiş teslim ayı içeren **değerlendirme** tarihleri; 3.320’nin paydası farklıdır. Overlay `verified-daily-overlay.json`, yeniden kurma arşivi `ams-contract-reconciliation-20261029/evidence.zip`.

**Kesinlik** — kesin.

**Kaynak** — `docs/archive/price-phase/AMS_CONTRACT_RECONCILIATION_20261029.md:3–39,41–49,58–81`, `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md:46–49`, `research/ams_contract_reconciliation.py:163–183`, `research/evidence/ams-contract-reconciliation-20261029.json:158–168`. [Private release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ams-contract-reconciliation-20261029).

**Claude nereye bakmalı** — Release `manifest.json` ve `evidence.zip` içindeki `verified-daily-overlay.json`, `day-reconciliation.csv`, bağımsız replay makbuzu. Overlay kısmi kimlik onarımı; çözümsüz tarih ve saat/vintage doğrulaması ayrı.

### 15. 2021 sonrası fiyat mührü

**Cevap** — Phase 1 için genel bir 2021 sonu fiyat mührü bulmadım. 2016–2023 ve görülmüş 2024+ incelemeler tarihsel olarak kullanıldı. 2024+ için sonraki seçim/tuning yasağı, verinin hiç görülmemesi anlamına gelmez. Phase 1’in gerçek ileri fiyat/risk kaydı ayrı, zamanında oluşan kilitli cohort. **15 Kasım Link 1** çalışmasında ise amaca özel mühür var: 2021-12-31 sonrası rapor olayları ve fiyat/getiriler bu zincirden dışarıda. Script fiyatı hesap öncesi `<= 2021-12-31` filtreler. T39 yalnız production finals ve USDA forecast’lerini açmıştı; fiyat mührü devam ediyor. Link 1 için 16 Kasım paketinde ayrı bir açma kaydı bulunmuyor. T39’un belgelenen açma süreci kullanıcı onayı ve push edilmiş ön kayıt istiyordu. Fiyat zincirinin açılması için T39’a dayanılmamalı. Phase 1’de daha önce okunan yılları bu çalışma için evrensel “unseen” diye sunmamak gerekir.

**Kesinlik** — kesin.

**Kaynak** — `README.md:8–16`, `docs/archive/MODEL_EXPERIMENTS.md:49`; Claude `docs/prereg/LINK1_EVENT_STUDY_PREREG_20261115.md:11–18,51–53`, `research/chain/link1_event_study.py:21,48,107`, `docs/prereg/SEALED_TEST_T39_PREREG_20261115.md:3–5,34–40`.

**Claude nereye bakmalı** — Link 1 ön kaydı, kod filtresi ve T39’un dar açma sınırı. Açılma tarihi/kararı iddiası için o kapsamı açıkça adlandıran ayrı ön kayıt ve kullanıcı izni.

## G · Kararlar ve bağlam

### 22. %5, %53, %55

**Cevap** — Eray, 16 Kasım 2026’daki bu görüşmede Codex’ten zor, başarılması kolay olmayan bir hedef koymasını istediğini ve sayısal başarı eşiklerini Codex’in belirlediğini bildirdi. Bu bilgi fikir sahipliği ve genel niyeti açıklıyor. Her bir kesin sayının seçimine ait nicel gerekçe ayrıca kayıtlarda bulunamadı.

21 Eylül `2eb44a0` sabitleri kodluyor. Fiyat MAE’sinde Naive’a karşı ≥%5, yön için T+1 ≥%53 ve T+5 ≥%55. V1 için 3/4 blok, daha sonraki sekiz yıllık protokol için 6/8 tutarlılık kapısı var. Bunlar proje tarafından seçilen pratik başarı kuralları. Eşikleri istatistiksel anlamlılık testi veya dış otorite standardı olarak sunmaya dayanak bulamadım.

**Kesinlik** — kesin. Değerler commit’ten; zor hedef talebi ve sayıları Codex’in belirlediği bilgisi doğrudan Eray’ın 16 Kasım 2026 beyanından. Sayıların tek tek nicel türetilişi bilinmiyor.

**Kaynak** — Eray’ın bu sohbetin 16 Kasım 2026 tarihli açıklaması; `2eb44a0:ml/src/cottonlens_ml/selection.py:5–7`; `docs/archive/MODEL_EXPERIMENTS.md:41–49`; `docs/archive/CONTINUOUS_RESEARCH.md:28`.

**Claude nereye bakmalı** — Görüşmede Eray’ın zor hedef istemesi ile Codex’in sayısal eşikleri belirlemesini ayrı katkılar olarak anlatabilir. Daha ayrıntılı seçim gerekçesi istenirse başlangıç sohbetine bakılmalı.

### 23. Eray mı, Codex mi?

**Cevap** — Eray’ın 16 Kasım 2026 beyanına göre zor ve başarılması kolay olmayan hedef talebi Eray’dan, %5 / %53 / %55 gibi somut başarı eşiklerini belirleme Codex’ten geldi. Diğer kararlar için kanıtla kurulabilen ayrım **onay/sınır** ve **uygulama/analiz** düzeyinde. Eray’ın 30 Ekim’de günlük konsolsuz Windows HAR görevini açıkça onayladığı, eski saatlik görevin kapalı kalacağı ve bütçe/kapsam sınırları belgeli. Kasım GPU kaydında Eray’ın A100’e bağlanamadığı, L4’e bağlandığı ve alternatif GPU istediği yazıyor. Codex tarafında dated plan, kod, kayıp/kol denetimi, frozen kaynak/receipt, ayrı forward runtime, Naive/EWMA benchmark değerlendirmesi ve app/araştırma teslimatı kanıtlanabilir. `2eb44a0` eşiklerin koda girişidir; fikir sahipliği Eray’ın sonraki açıklamasına dayanır. V1 LSTM mimarisi, HAR secondary hipotezi veya her özellik/deney için “ilk fikir kimindi?” tek tek **bilmiyorum**. Bir kararın kullanıcının onayından geçmesiyle ilk önerinin kimden geldiği farklı olaylardır.

**Kesinlik** — bilmiyorum. Eşiklerin fikir sahipliği Eray’ın doğrudan beyanıyla açıklandı, diğer bütün kararlar için eksiksiz fikir sahipliği haritası yok.

**Kaynak** — Eray’ın bu sohbetin 16 Kasım 2026 tarihli açıklaması; `docs/archive/price-phase/CT_FORWARD_DAILY_20261030.md:3–19,32–58`, `docs/archive/price-phase/DATA_GPU_PROGRESS_20261106.md:165,181`, ana `docs/CLAUDE_PROJECT_HANDOFF_20261001.md:25–37`, ana `docs/archive/phase2-notes/CLAUDE_HANDOFF.md:95–113`, Kasım 6 Codex rollout’unda kullanıcı mesajı (planı yürütmeye devam, dosya satırı 991).

**Claude nereye bakmalı** — `C:/Users/Eray/.codex/sessions/2026/11/06/rollout-2026-11-06T22-27-29-01a1ab15-35f6-70c3-a772-07001ee27f19_01a1b254-9178-7422-872c-f560b7175c92.jsonl`, özellikle kullanıcı mesajları; Eylül için özgün sohbet dışa aktarımı veya Eray’ın kendi plan notu gerekir.

### 24. Private arşiv ve diğer worktree’ler

**Cevap** — En önemli ek malzeme **ham/makinece okunur kanıt**: v1 ve R2 Colab TXT/JSON/ZIP, v1 Keras state ve eğrileri, R2 checkpoint makbuzları, sprint tamamlama makbuzları, market-v1 20.360 completion ve ayrı raporu, Round A karar vektörleri ve kurtarılmış kısmi makbuzlar. Bunlar bazı soruları Git özetinden daha güçlü yanıtlıyor. İki worktree’nin farkı da önemli. `445d1fe` açık PR #77’nin kaynak/defect audit ve median replay kapsamını, `5b22559` end-to-end audit’in o anki kapsamını gösterir. Devirde median restore, 224 üyeli source/L4 evidence ve özel CatBoost makbuzu ayrıca doğrulanmış. Eski kayıp payload referansları var. İncelenen arşivlerde tarihsel price STOP, risk forward durumu veya DG-008’i tek başına tersine çevirecek yeni doğrulanmış sonuç görmedim. Tüm özel asset payload’larının kapsamlı taraması yapılmadı.

**Kesinlik** — kesin. Bulunan ek kanıtlar ve sınırları dosyadan doğrulandı.

**Kaynak** — Ana `docs/archive/phase2-notes/CLAUDE_HANDOFF.md:9–49,115–191`, `docs/archive/price-phase/GITHUB_DELIVERY_20261105.md:6–24`, `research/README.md:672`, Q1–Q2/Q6–Q7/Q12’deki yollar, `5b22559` ve `445d1fe`. [Median release’i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/price-only-median-result-20261105) · [Source gate/L4 release’i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/source-gate-l4-audit-20261106) · [Drive preservation](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/drive-preservation-20261011).

**Claude nereye bakmalı** — Önce ana checkout `output/claude-audit/drive/`, `output/claude-preparation-20261112/` ve `CottonLensAI Colab/CottonLensAI/`; sonra `C:/Users/Eray/.codex/worktrees/naive-mae-20261105/CottonLensAI/output/data-gpu-audit/` ve `C:/Users/Eray/.codex/worktrees/repository-curation/CottonLensAI/docs/END_TO_END_AUDIT.md`. Farklı restore kopyalarını fit toplamına iki kez ekleme.

### 25. Görüşme için 10 belge

**Cevap** — Önerdiğim okuma sırası:

1. `docs/archive/phase2-notes/CLAUDE_HANDOFF.md` — doğru commit, yerel ve private kanıt haritası.
2. `docs/archive/MODEL_EXPERIMENTS.md` — v1 sonuç, LSTM ve dört fold, v2 düzeltmeleri.
3. `docs/archive/DAY1_R2_EVIDENCE_REVIEW.md` — düzeltilmiş LSTM eğrisi, dataset ve ölçüm sınırı.
4. `docs/archive/CONTINUOUS_RESEARCH.md` — Round A–E bütçeleri, protokol, release ve replay.
5. `docs/archive/price-phase/NAIVE_MAE_AUDIT_20261105.md` — Naive neden yenilmedi sorusunun kanıt envanteri ve eski LSTM düzeltmesi.
6. `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md` — son fiyat hipotezi, bağımsız replay ve STOP.
7. `docs/archive/price-phase/ROOT_CAUSE_REVIEW_20261106.md` — tek kök neden bulunmadığının savunulabilir sentezi.
8. `docs/archive/price-phase/AMS_CONTRACT_RECONCILIATION_20261029.md` — kaynak kimliği yanlışının fit ve evaluasyona etkisi.
9. `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md` — DXY ana kolu, own-level ikincil QLIKE sonucu ve tail yoğunlaşması.
10. `docs/archive/price-phase/END_TO_END_AUDIT.md` — araştırma ve runtime sınırları, sonda v1/v2 forward yayın tablosu.

**Kesinlik** — kesin. Belge yolları ve kapsam dosyadan. Sıralama görüşme amaçlı editoryal seçim.

**Kaynak** — Numaralı belge yolları ve `docs/README.md` indeksi. İlk belge ana checkout veya Claude’a bırakılan kopyada.

**Claude nereye bakmalı** — Her belgeye başlıkta verilen worktree sürümünden. İleri operasyon makbuzu Q16–Q18’de, demo komutu Q19’da.

**Özet** — 27 soru: 24 kesin · 0 hafızadan · 3 bilmiyorum. “Kesin” işaretli yanıtlardaki açık alt sorular ilgili bölümde ayrıca belirtilmiştir.
