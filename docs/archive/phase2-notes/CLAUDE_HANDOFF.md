# CottonLensAI: bağımsız araştırma için kaynak haritası

Bu belge 12 Kasım 2026'da, yerel dosyalar ve GitHub'ın erişilebilen durumu
incelenerek hazırlanmıştır. Amaç özgün kanıtlara ulaşmaktır; aşağıdaki geçmiş
kararlar ve yorumlar bağımsız araştırmanın sonucu olarak kabul edilmemelidir.
Kapsamlı bir arşiv bütünlüğü veya tüm yerel dosyaların uzakta bulunduğu iddia
edilmemektedir.

## Önce doğru sürümü seçin ve yerel çalışmaları koruyun

- Ana yerel depo: `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI`.
  Bu çalışma ağacı `codex/free-data-research-workbench-20261001`, `470f1e5`
  üzerinde ve kendi upstream dalından dört commit gerideydi. Çok sayıda
  değiştirilmiş ve izlenmeyen kod, test, notebook ve belge vardır; `git status
  --short` ile güncel listeyi inceleyin. Bunlar henüz kaydedilmemiştir ve
  temizleme, reset, checkout veya toplu commit sırasında kaybolabilir. Bu
  belgenin bulunması, diğer yerel değişikliklerin de Git'te olduğu anlamına
  gelmez.
- Yerel `main` ve `origin/main` referansı inceleme sırasında `943944d` idi.
  Uzak durumu kullanmadan önce `git fetch` ve `git log` ile ayrıca doğrulayın;
  bu dosyanın hazırlandığı çalışma ağacı `main` değildir.
- En yeni araştırma düzeltmeleri [taslak PR #77](https://github.com/ErayKulkizaga/CottonLensAI/pull/77)
  dalında, `445d1fe` commit'indedir; `main` ile aynı değildir. PR bu incelemede
  açıktı ve son commit'in backend, frontend, ML ve Compose kontrolleri GitHub'da
  başarılı görünüyordu. Bu durum zamanla değişebilir. PR geçmişi ve diff'i de
  kanıtın bir parçasıdır.
- Yerel araştırma worktree'si:
  `C:/Users/Eray/.codex/worktrees/naive-mae-20261105/CottonLensAI`.
  `docs/archive/price-phase/DATA_GPU_EXECUTION_PLAN_20261106.md` üzerinde yerel değişiklik ve
  izlenmeyen `docs/COTCAST_REVIEW_20261106.md` vardır. Bunları ayrıca okuyun;
  PR'ın veya yeni bir worktree'nin içinde otomatik bulunmazlar.
- Bir başka yerel worktree:
  `C:/Users/Eray/.codex/worktrees/repository-curation/CottonLensAI`, `5b22559`.
  Taşınma nedeniyle bozulan iki worktree bağlantısı 12 Kasım'da `git worktree
  repair` ile onarıldı; Git durumları onarım öncesiyle aynıydı. İkinci
  worktree'deki izlenmeyen `tmp/pdfs/ct-manual-risk-20261030/` iki PNG
  önizlemesi içerir; dosyalar korunmuştur.

Hazırlanmış ayrı inceleme alanı:
`C:/Users/Eray/SEZAR/PROJELER/CottonLensAI-claude-review`, yerel
`codex/claude-handoff` dalı, `445d1fe` tabanı. Devir belgesi commit onayı
bekleyen tek staged dosyadır. Bu Windows kullanıcısıyla bu klasörden `gh`
üzerinden private depo ve örnek release metadata erişimi doğrulandı; Claude
uygulamasının oturumu ve kendi ortamı ayrıca doğrulanmamıştır.
Yerel kirli dosyaları veya aşağıdaki ignore edilmiş verileri otomatik taşımaz.
Ana çalışma kopyasının kod/belge değişiklikleri ve araştırma worktree'sindeki
iki belge için doğrulanmış, yerel ve ignore edilmiş kurtarma kopyaları
`output/claude-preparation-20261112/` altındadır. Kişisel `opencode.json`,
gizli dosyalar ve toplu veri dahil edilmedi. Bu kopyaları tamamlanmış uzak
yedek saymayın. Yerel araştırma notlarını özgün konumlarında da inceleyin.

## Araştırma kaydının giriş noktaları

Aşağıdaki yollar, aksi belirtilmedikçe PR #77'nin `445d1fe` ağacına aittir;
eski ana çalışma ağacında tamamı bulunmayabilir. Bunları PR web arayüzünden,
araştırma worktree'sinden veya yeni bir `445d1fe` worktree'sinden okuyun.

- `research/README.md`, `research/registry.json`, `research/trials.json`:
  deney, tarif ve iş kayıtları; `research/evidence/` ön kayıtlar, sonuçlar,
  teslim ve doğrulama makbuzları. Sicil kopya sayılarını bağımsız deney sayısı
  olarak yorumlamayın. `python ml/history.py validate` sicil kontrolüdür;
  `python ml/history.py check --help` kapsamlı arama seçeneklerini gösterir.
- `docs/README.md`, `docs/archive/price-phase/STATUS.md`, `docs/archive/README_HISTORY_20261105.md`:
  belge indeksi, tarihli durum ve geçmiş. Durum belgeleri yazıldıkları tarihin
  kayıtlarıdır; tarihsel önerileri güncel izin veya sonuç olarak varsaymayın.
- `docs/archive/price-phase/DATA_GPU_EXECUTION_PLAN_20261106.md`,
  `docs/archive/price-phase/DATA_GPU_PROGRESS_20261106.md`, `docs/archive/price-phase/G2_COVERAGE_20261107.md` ve
  `research/data-gpu-defects.json`: veri kabulü, sızıntı kontrolleri, açık
  kanıt boşlukları ve düzeltmeler. Yerel plan değişikliğini yukarıdaki worktree
  kopyasında ayrıca inceleyin.
- `docs/archive/price-phase/NAIVE_MAE_PLAN_20261105.md`, `docs/archive/price-phase/NAIVE_MAE_AUDIT_20261105.md`,
  `docs/archive/price-phase/PRICE_ONLY_MEDIAN_RESULT_20261105.md` ve
  `research/evidence/price-only-median-result-20261105.json`: fiyat hedefleri,
  sabit protokol, özgün sonuç ve arşiv kimlikleri. Makbuz sabit T+1 deneyinde
  338 piyasa ve 4 sentetik fit, 2.006 origin, Quantile için yaklaşık −%0,0825
  seçili MAE kazancı ve STOP kaydeder. Bu sayı, ilgili kayıt ve bağımsız
  hesapla yeniden incelenebilir; bütün yöntemler veya gelecek dönemler hakkında
  genelleme değildir.
- `docs/archive/price-phase/SOURCE_GATE_RECHECK_20261106.md`,
  `research/evidence/ams-required-cell-readiness-20261029.json` ve
  `research/data-gpu-defects.json`: named-contract etiket, kaynak ve bağımlılık
  sorunları. Kayıtlardaki 217 değerlendirme tarihi ile 99.523 potansiyel
  UNKNOWN hücre farklı kapsamlı ölçülerdir; ikincisi gerçek fiyat hatası sayısı
  değildir. Resmi kontrat yaşam döngüsü, yayın zamanı/vintage ve lisans kanıtı
  için ICE yanıtı beklenmektedir; bu belge bir yanıtı doğrulamamıştır.
- Diğer yöntem ve sonuçlar için sicil ve belge indeksinden profil adına göre
  ilerleyin. Örnek kayıtlar: `docs/archive/price-phase/CT_LEVEL_DXY_CONTROL_20261030.md`,
  `docs/archive/price-phase/CT_SHOCK_FORECAST_CONTROL_20261029.md`,
  `docs/archive/price-phase/CT_INFERENCE_CALIBRATION_20261031.md` ve
  `docs/archive/price-phase/CT_PROSPECTIVE_READINESS_20261030.md`. Sentetik öğrenme ve operasyonel
  doğrulama, gerçek piyasa üstünlüğüyle aynı kanıt değildir. Tarihsel LSTM
  karşılaştırmasının eksik orijinal ortak-küme tahminleri için hata sicilindeki
  DG-008'e bakın; eksik kanıtı var kabul etmeyin.

## Kod, protokoller ve testler

- `ml/src/cottonlens_ml/research/`: araştırma motoru, veri kabulü (`admission.py`),
  ledger/checkpoint, karar cache'i, değerlendirme, native tekrar üretim ve
  model yolları. `ml/tests/` ve `research/tests/` ilgili testleri içerir.
- `research/verify_naive_mae.py`: fiyat kaybı ve ortak origin için bağımsız
  hesaplayıcı; `research/storage.py`: arşiv paketleme, doğrulama ve geri
  yükleme araçları. İlgili makbuz ve manifest ile kapsamlarını birlikte okuyun.
- `ml/notebooks/` ve `ml/` içindeki Colab giriş noktaları GPU altyapısını;
  `backend/`, `frontend/` ve `runtime/` uygulama tarafını içerir. Test komutları
  ve ortam sınırları PR ağacındaki `AGENTS.md` ve `.github/workflows/ci.yml`
  dosyalarında kayıtlıdır. CI sentetik kontrolleri piyasa sonucu olarak sunmaz.
- Aktif kullanıcı talimatı: incelemeye güvenli ve düşük maliyetli başlayın;
  büyük değişiklikler, yeni eğitim deneyleri, maliyetli işlemler ve protokol
  değişiklikleri için önce kullanıcı onayı alın. Onaylı değişiklikler için ayrı
  Git dalı kullanın. Üretim modeli, API ve otomasyonlarda ayrıca yetki gerekir.
  Bağımsız inceleme sırasında veri erişilebilirliği,
  karar zamanı, hedef olgunluğu, split/purge, ortak cohort ve kanıt kimliğini
  önce ilgili deneyin özgün sözleşmesiyle karşılaştırın.

## Yerel veri ve model: yeni worktree'ye taşınmaz

Bu bölümdeki `output/` referansları ana checkout altındadır:
`C:/Users/Eray/SEZAR/PROJELER/CottonLensAI/output/`. Bunları inceleme
worktree'sinin kendi `output/` klasöründe aramayın.

Ana deponun `.gitignore` dosyası `output/`, `ml/data/`, `ml/checkpoints/`,
`runtime/artifacts/*`, ZIP ve Parquet dosyalarını dışlar. Yeni worktree bu
dosyaları ve yerel değişiklikleri içermez. Yerel yolların varlığı, içeriğin
tamlığı veya kullanılabilirliği için tek başına kanıt değildir.

- `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI/output/`: çok sayıda kaynak,
  deney ve rapor klasörü; örnekler `full-year/experiments`,
  `usda-ams-2016-2019`, `wasde-vintage`, `power-weather-2010-2023` ve
  `own-level-forward-v2-first-observed`. Kapsamları tek tek doğrulanmadı.
- Ana deponun `runtime/artifacts/current/` klasöründe model, manifest, checksum,
  forecast ve Parquet dosyaları görüldü. Bunların güncelliği, lisansı ve her
  dosyanın hash'i bu incelemede doğrulanmadı. `ml/data` ve `ml/checkpoints`
  için dolu bir yerel içerik gösterilemedi; başka konumlar olabilir.
- Araştırma worktree'sindeki
  `output/data-gpu-audit/median-independent-restore-20261106/evidence/experiment/`
  içinde `history.parquet`, `ledger/`, `reference/`, raporlar ve kaynak snapshot'ı
  görüldü. Bunlar o worktree'nin yerel, ignore edilmiş dosyalarıdır. Aynı
  worktree'nin `output/data-gpu-audit/` altında L4, kaynak ve restore kayıtları
  da bulunmaktadır. Önce ilgili makbuzlara ve hash'lere bakın.
- Kaynak envanterinde AMS, FAS, NASS, WASDE, CFTC, ECB/FX, fiyat ve hava verisi
  çalışmaları bulunur; sağlayıcı, alan, kontrat, lisans ve point-in-time
  geçerliliği deneyden deneye değişir. `research/registry.json`, ilgili
  ön kayıt, manifest ve kaynak denetimi asıl kapsamı belirler.

## Private GitHub arşivleri ve erişim

Depo `ErayKulkizaga/CottonLensAI` private'tır. Bu makinede `gh auth status`
başarılıydı ve 111 release listelendi (`gh release list -R
ErayKulkizaga/CottonLensAI --limit 200`). Claude'un kullanacağı ortamın ayrıca
yetkilendirilmesi gerekir; bu makinedeki oturum başka ortamda erişim garantisi
vermez. Kimlik bilgilerini belgeye veya Git'e koymayın.

- [Price-only median result release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/price-only-median-result-20261105):
  `evidence.zip`, `manifest.json`, bağımsız doğrulama ve replay makbuzları.
  Bu incelemede manifest ve ZIP taze indirildi; manifest SHA-256 değeri
  `7948cef7309f71174519eb8cb1562d3da3715fd0d03b2defe80c54e561626224`
  ile eşleşti. `research/storage.py restore`, 2.085 üyeyi doğrulayıp ayrı bir
  yerel klasöre geri yükledi; model deserialize veya fit edilmedi.
- [Source gate / L4 audit release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/source-gate-l4-audit-20261106):
  manifest, kaynak/lifecycle incelemeleri, GPU restart/OOM/session ve bağımsız
  doğrulama paketleri. Ana `evidence.zip` ve manifest taze indirilerek 224 üye
  ayrı klasöre doğrulanıp geri yüklendi; diğer bütün asset'ler yeniden
  indirilmedi. İki geri yükleme `output/claude-preparation-20261112/` altında.
- Diğer release'leri profil veya sicil kaydındaki tag ile bulun:
  `gh release view <tag> -R ErayKulkizaga/CottonLensAI`;
  `gh release download <tag> -R ErayKulkizaga/CottonLensAI --dir <ayrı-klasör>`.
  Büyük dosyaları indirmeden kapsam, lisans ve depolama gereksinimini kontrol
  edin. `research/storage.py --help` ve release'in kendi manifest/makbuzu ile
  hash ve restore işlemini doğrulayın; eski dosyaların üstüne yazmayın.

Özgün 5 Kasım kaydı eski TF-placement ve on-call çalışmaları için 22.299 eksik
yerel payload referansı bildirir. 12 Kasım karşılaştırmasında yerel TF-placement
makbuzlarının 7.331'i incelendi: 21.993 referansın hiçbiri tek
`coherent-drive-v1-inventory.json` envanterindeki tam yol ve hash ile eşleşmedi;
yerel ledger altında da hash'i doğrulanan kopya bulunmadı. Ayrıca
111 release içindeki manifest adını taşıyan 139 JSON dosyası indirildi; desteklenen
`files`/`members` yol ve SHA-256 alanlarında aynı yol veya aynı hash eşleşmesi
bulunmadı. Bu metadata taraması farklı biçimli alanları, manifest dışı asset
içeriklerini ve on-call makbuzlarını kapsamaz; tüm dosyaların kayıp olduğunu
kanıtlamaz. Yerel karşılaştırma: `output/claude-preparation-20261112/` altındaki
`all-manifest-tf-comparison-v2.json`. Ayrı `b93f...` CatBoost
makbuzu ve üç native dosyası `drive-preservation-20261011` release'inden taze
indirilip hash ile doğrulandı; bu makbuz yukarıdaki yerel kümede yoktur.
On-call kümesinde 222 yerel makbuz incelendi; mevcut 360 dosyanın hash'i
makbuzlarıyla eşleşti. 102 makbuzun 306 dosya yolu yerelde eksikti. Bunların
45 referansı aynı boş `curves.json` içeriğinin hash'iyle desteklenen manifest
alanlarında eşleşir; diğer 261 referans için hash adayı bulunmadı. Bu eşleşme,
özgün bağlamın veya model/adapter dosyalarının geri kazanıldığı anlamına gelmez.
Karşılaştırma özetleri ve indirilen manifestler yerel
`output/claude-preparation-20261112/` altındadır. Kayıp LSTM tahminleri ve ICE
kaynak/vintage kanıtı da açık görünür.

111 release metadata'sı listelendi; `research/evidence/*.json` içindeki literal
103 release-tag bağlantısının tamamı bulundu. Metadata/etiket varlığı tüm
asset'lerin sağlamlığını veya lisansını kanıtlamaz. Bu Windows bilgisayarında Claude masaüstü uygulaması kurulu (5.1231.2.0);
ayrı `claude` terminal komutu PATH'te ve kontrol edilen yaygın konumlarda
bulunmadı. Resmi Claude belgelerine göre masaüstündeki Code sekmesi kendi
Claude Code bileşenini kullanır. Kullanıcının uygulama oturumu, Code sekmesinin
kullanılabilirliği ve private GitHub erişimi henüz bu sekmede doğrulanmadı. Eksiklikler belli bir bilimsel sonucu zorunlu kılmaz.
