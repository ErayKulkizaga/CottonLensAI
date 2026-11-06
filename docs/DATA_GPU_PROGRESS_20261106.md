# Veri/GPU uygulama takip kaydı

Yetkili kapsam: [onaylanan plan](DATA_GPU_EXECUTION_PLAN_20261106.md).
Bu takip kaydı tamamlanma iddiası değildir. Önceki bilimsel kayıtlar değişmez.

| Teslimat | Durum | Mevcut kanıt / kalan iş |
|---|---|---|
| A: Hata ve kapsam sicili | DEVAM | [Hata sicili](../research/data-gpu-defects.json);239 deney kaydı,139/106/28 ve217scored lifecycle sorunu yeniden doğrulandı; bütün gereksinimlerin kapanışı henüz kanıtlanmadı |
| B: Veri sözleşmesi | DEVAM | Role-aware dependency admission çekirdeği eklendi; source-specific doğrulama ve gerçek hücre matrisi bekliyor |
| C: Onarım ve etki yayılımı | DEVAM |139/106/28yeniden doğrulandı; katı kapsam0hazır, tam kaynak onarımı eksik |
| D: Zorunlu eğitim kapısı | DEVAM | Ledger/model/backend/legacy fit ve doğrudan QLIKE kernel kimlik kontrolü; bütün seçim/preprocessing yollarının kapsam denetimi açık |
| D: Sızıntı testi | DEVAM | Kabul ve hata enjeksiyonları;5fiyat feature'ı169cutoff prefix/future testi geçti. Bütün model/seçim yollarının prediction invariance denetimi eksik |
| E: Bağımsız değerlendirme | BEKLİYOR | Eski verifier'lar mevcut; yeni sözleşmeye kapsam eşlemesi yapılacak |
| F: GPU yürütme/restore | KISMİ | Kullanıcının onayladığı L4 sentetik/native Adam replay ve uzak arşiv restore geçti; zorla kesinti/OOM/worker çakışması eksik |
| G: Kampanya manifesti/sonuç | G1/G2/G3 BEKLİYOR | Veri kabulü olmadan yeni piyasa fit'i yok |
| GitHub teslimatı | DEVAM | Draft PR77 ve checksum bağlı private Release mevcut;41ffe0e PR CI başarılı. Sonraki commit'ler ayrıca doğrulanacak |
| İleri doğrulama kararı | BEKLİYOR | Aday ve geçerli kampanya sonucu oluşmadan karar yok |

Aşağıdaki bölümler tarih sıralı çalışma kaydıdır; eski anlık durumlar yukarıdaki
güncel tablonun yerine geçmez.

## İlk doğrulama

`PYTHONPATH=ml/src python -m pytest ml/tests/test_admission.py -q`:16 passed.
İki yeni dosyada Ruff geçti. Bu hafif standart-kütüphane testleri host Python3.13
ile çalıştı; pinned Colab/Python3.12 entegrasyonu henüz denenmedi.

Kapsanan hatalar: timezone yokluğu, bilinmeyen/geç feature erişimi, eğitim
etiketinin strict fit sınırında olgunlaşmaması, belirsiz ata, eksik kimlik/vintage/
tanık, değişmiş payload/tarif/makbuz, eksik/döngüsel/tekrarlı bağımlılık, açık
kritik kusur ve sessiz rol/kapsam düşürme. Geçerli örnek ve hata sonrası eski
geçerli örneğin yeniden geçmesi kontrol edildi. Gerçek piyasa eğitimi0.

Hash ve VERIFIED alanı tek başına kaynak doğruluğu değildir. Bu ilk çekirdek,
source-specific tanık doğrulamasının yerine kullanılamaz; yeni eğitim henüz
bu API üzerinden otomatik yetkilendirilmemiştir.

## İkinci uygulama turu

- Kısmi tabloda ay/sıra karşılaştırmasından quote yokluğu çıkarımı kaldırıldı.
  Tam olmayan tabloda görünmeyen kontrat UNKNOWN kalır. Eski frozen code ve
  sonuçlar değişmedi; güncel kaynak readiness sayılarının yeniden hesaplanması
  gerekiyor. Bu değişiklik eski kaynak kusurlarını çözmüş sayılmaz.
- Yeni admitted ledger namespace'inde makbuz eksikliği, yanlış iş tarifi,
  değişen kaynak ve cache sonrası kontrol test edildi. Budget/staging callback'i
  kaynağı değiştirirse işlem compute'e girmeden reddediliyor.
- Eğitim özelliği ile inference özelliği ayrıldı: fit ile decision arasındaki
  erişim yalnız inference için kabul edilir. Etiket olgunluğu ayrıca denetlenir.
- `test_admission`, `test_ams_cell_readiness`, `test_ams_contract_reconciliation`:
  toplam34 passed; değişen5Python dosyasında Ruff geçti. Sicil valid239/569/32934.
- Henüz bütün eski eğitim yolları kabul politikasına taşınmadı; yeni kampanya
  çalıştırılmadı. Kaynak-specific tanık doğrulaması ve gerçek dosya/readiness
  restore'u sonraki uygulama adımlarıdır.

Ek bulgu: Kaynak tablosu bulunmayan günler default `complete=True` ile yokluk
kanıtı gibi işleniyordu. Ana readiness default'u False yapıldı. Bağımsız
doğrulayıcıdaki aynı varsayım ve ay/sıra çıkarımı da ayrı literal fonksiyonla
kaldırıldı. Yeni kuralla eski raporların sayıları henüz replay edilmedi;
arşivdeki frozen verifier eski raporu açıklamak için korunur. Güncel testler:
admission/checkpoints/cell-readiness/contract-reconciliation toplam40 passed;
değişen6Python dosyasında Ruff geçti. Sicil dosyaları değiştirilmedi.

## Gerçek arşiv ve kaynak hesapları

[Kaynak yeniden denetimi](SOURCE_GATE_RECHECK_20261106.md):401dosya taze restore;
frozen-source11çıktı byte eşit;139/106/28feature-etki sayıları yeniden doğrulandı.
Katı bağımlılık hesabını ikinci verifier doğruladı:662yerine0tam hazır origin,
174işin tümünde hazır olmayan train/test. Bilinmeyen hücre üst sınırı40450;
gerçek fiyat hata sayısı değildir. G1geçmedi. Çıktı hash'leri
research/evidence/source-gate-review-20261106.json içinde.

[Kaynak talep/örnek kabul belgesi](DATA_SOURCE_REQUEST_20261106.md) hazır;
resmi ürün kapsamları incelendi, fiyat/teklif alınmadı ve iletişim gönderilmedi.
Colab erişim envanterinde mevcut browser tab/runtime yoktu; canlı A100 henüz
doğrulanmadı. Bu, hesabın/A100erişiminin bulunmadığı anlamına gelmez.

## Eğitim bağlantısı ve A100 hazırlığı

Ledger artık yalnız policy1namespace'inde değil; recipe içeren veya profile/
source_idtaşıyan bütün yeni compute işlerinde ExecutionAdmission istiyor.
Eski tamamlanmış kayıtlar identity/specification/hash eşleşmesiyle okunuyor.
Experiment, varsa immutable admission/manifest.json ve receipt.json yüklüyor;
eksik çifti reddediyor. CLI ml/admission.py hesap raporunu frozen makbuzla üretir.
Bu mekanizma source-specific veri kabulünün yerini tutmaz; doğrudan model ve
legacy pipeline girişlerini tam kapatma henüz bitmedi.

Python3.12'de research/review/naive/admission entegrasyon79test geçti. Yeni
zorunluluk nedeniyle generated-fixture testleri açık sentetik kabul adapter'ına
taşındı; üretim paketi bu adapter'ı içermez. Geniş MLsuite çalışıyor, yeni
kuralla uyumsuz test/çağrılar var; henüz bütün testler geçti iddiası yok.

Colab tarayıcı hesabının oturumu açılmış olduğu görüldü. Google resmi CLI'si
Windows'u desteklemediğini bildiriyor; mevcut tarayıcı yolu kullanılabilir.
ml/a100_preflight.py yalnız üretilmiş veride3train_on_batch çağrısı, gerçekA100,
optimizer-state resume ve native reload kontrolü için hazırlandı. Henüz Colab'da
çalıştırılmadı/G3geçmedi. Hostta3ret testi ve22admission testi geçti; bunlar
GPUçalışma kanıtı değildir. Piyasa fit'i0.

Geniş suite ilk çalışmada886passed/9failed/3errors/3skipped ile12failure
sınırında durdu. Hataların tamamı yeni zorunlu kabulün generated/stub işleri
de kapsamasıydı. İlgili testler yalnız opt-in `admit_generated_jobs` fixture'ına
taşındı; bu gerçek graph validator'ı çalıştırır ve eğitim paketine dahil değildir.
Başarısız gruptaki14test düzeltme sonrası geçti. Tüm suite henüz yeşil sayılmaz.

Lifecycle bağımsız yeniden sayımı tamamlandı:
research/evidence/lifecycle-recheck-20261106.json.218history ve218evaluation
slotunda geçmiş kontrat var;217'si eligible/scored hedef.2020-06-11origininde
kontrat hedefte bulunmadığından eski named_eligible=False.217sayısı2019'da106,
2020'de111olarak yeniden üretildi. Literal annual ikinci sıra kimlikleri aynı;
bu ay-invariant kontrolü exact expiry/PIT doğrulaması değildir.

## Gerçek girdi kimliği ve canlı A100 tahsis isteği

ExecutionAdmission.training_scope ile modelin gerçek history/train/validation/
test tabloları, recipe ve iterations hash'ine bağlanan context kontrolü eklendi.
Bu API'nin bütün model girişlerine yerleştirilmesi henüz bitmedi. Bağlam dışı
çağrı, altı ayrı girdi/tarif değişikliği ve bağlamdan çıkışta yetkinin sıfırlanması
sentetik test edildi. Admission+A100host ret testleri toplam32passed/Ruff geçti.

Yeni Colab teknik defteri:
https://colab.research.google.com/drive/1k2qeXZ_QMUvwekam57zMgsSor4LRJhg2
A100 seçildi ve yalnız nvidia-smi donanım sorgusu sıraya alındı. Son gözlem
"Çalışma zamanı ayrılıyor/Bağlanıyor"; donanım çıktısı yok. Aynı defter/tahsis
izlenecek, bekleme nedeniyle ikinci runtime başlatılmayacak. Model eğitimi0.
Tam MLr4çalışması mevcut session76541ile sürüyor; henüz nihai test sonucu yok.

## Kullanıcı yönlendirmesi: Colab kullanıcıda

Kullanıcı Colab işlemlerini kendisinin yürüteceğini belirtti. Tarayıcı üzerinden
Colab yönetimi durduruldu. Mevcut teknik defterin sonucu/A100tahsisi doğrulanmış
sayılmıyor. Diğer mühendislik, veri, test ve teslim işleri asistanda devam eder.

research/build_a100_handoff.py142kod-only dosyalı checksum bağlı kaynak ZIP ve
CottonLens_A100_Preflight.ipynb üretir. İlk paket output/a100-handoff-20261106
altında; notebook ve ZIP bağlantıları kullanıcıya verildi. Notebook4kod hücresi
ile Drive mount, whitelist/hash kontrolü, mevcut pinned Colab kurulumu, yalnız
sentetik teknik probe ve indirilebilir sonuç ZIP/checksum üretir. Kullanıcı
sonuç ZIP ve.sha256.jsondosyasını döndürecek. Bu sırada G1/G2çalışmaları sürer.
Yerelde142üye hash'i, path güvenliği, key-pattern taraması ve hücre AST'leri
kontrol edildi. Notebook/ZIP henüz Colab'da çalıştırılmış sayılmaz.

MLr4sonucu1297passed/5failed/3skipped. Kalan5generated-fixture testi yeni kabul
kapısına taşındı; ayrı çalışmada5passed. Sonraki üretim-model context bağlantısı
nedeniyle bu iki çalışma birleştirilip 'tam suite yeşil' ilan edilmedi.
Model-giriş kontrolü ve native sentetik roundtrip grubunda38passed/Ruff geçti.
Legacy doğrudan backend/pipeline yollarının tamamlanması hâlâ gerekiyor.

## Doğrudan backend denetimi

Statistical ve volatility fit_predict girişlerine kabul kontrolü eklendi.
Üst modelden gelen doğrulanmış history/train/test/recipe kimliği backend'de
tekrar karşılaştırılıyor. Doğrudan backend çağrısı da kendi tam job kimliğine
bağlanmak zorunda. Scope çıkışında hem iş hem backend yetkisi sıfırlanır.
35admission/statistical test ve değişen6Python dosyasında Ruff geçti.
Legacy training.py public pipeline ve özel HAR kernel çağrılarının denetimi
hâlâ açık; bu testler bütün girişlerin kapandığını kanıtlamaz.

Colab kaynak paketi ilk verilen hash'te korundu. Build script'i değişmiş
kaynakla aynı teslimat ZIP'ini ezmeyi reddediyor; yeni kaynak için yeni namespace
gerekir. Kullanıcıya verilen teknik paket sonraki mühendislik değişiklikleriyle
sessizce güncellenmez. G3sonucu kullanıcıdan bekleniyor; yerel işler sürüyor.

## Donanım kararı güncellendi

Kullanıcı A100'e bağlanamadığını, L4'e bağlandığını bildirdi ve alternatif GPU istedi. Probe artık A100/L4/G4 cihazını açık parametreyle seçer; sessiz fallback yoktur. G4 ancak gerçek RTX PRO 6000 tanığı ve framework kontrolüyle kabul edilir. Yeni L4 teslimatı output/gpu-handoff-l4-20261106 altında; eski A100 paketi korunur. Paketler karıştırılmamalı. Yerel 5 cihaz-ret/namespace testi ve Ruff geçti; bunlar gerçek GPU çalışma kanıtı değildir.

## Drive teslimatı

Kullanıcının açık isteğiyle L4 kaynak ZIP'i, source-manifest.json ve elle ZIP yüklemesi istemeyen CottonLens_L4_Drive.ipynb Drive'a yüklendi. Klasör: https://drive.google.com/drive/u/0/folders/1rtMBiFl2lMBR2DVCB0Ez5UlB75j5uqy2 . Drive arayüzünde3yükleme tamamlandı ve3dosya görüldü. Notebook kaynağı /content/drive/MyDrive/CottonLensAI/GPU_Handoff_L4_20261106 konumundan kendi hash'iyle okur. Uzak dosyalar ayrıca taze indirilip hash doğrulanmış sayılmıyor; notebook çalışırken kaynak hash kontrolü yapılacak. Colab çalıştırma kullanıcıda kalıyor.

Legacy pipeline yeni estimator ve preprocessing fit noktalarına tam fit kimliği denetimi eklendi; cached XGBoost okuyuşu yeni fit yetkisi gerektirmeden sınandı.42training-wiring/admission test ve Ruff geçti. Bu, bütün planın veya bütün source-specific kabulün tamamlandığı anlamına gelmez.

## Piyasa kaynağı için çapraz rapor kontrolü

Admission manifest artık market/synthetic amacını açıkça belirtir. Market leaf, hash'e bağlı source-verification raporundaki alan/sembol/kontrat/seans/vintage/erişim zamanı ile eşleşmek zorunda. Futures için listed_on <= session <= last_trade_on kontrol edilir; eksik tarih reddedilir. Raporun alan/kimlik/takvim/PIT/lifecycle kontrol durumları tam olmalı. Rapor okunurken hash tekrar kontrol edilir. Bu bir tutarlılık katmanıdır; raporun güvenilir yayıncıdan geldiğini imzalayan veya ilk-vintage kanıtını kendiliğinden yaratan bir sistem değildir. Source-specific doğrulama hâlâ gerekli ve gerçek named-risk verisi için G1 hâlâ kapalıdır.

57admission/legacy-wiring/native-synthetic test geçti; değişen3Python dosyasında Ruff geçti. Yeni market-shaped rapor testleri yalnız üretilmiş şema örnekleridir, gerçek piyasa kabulü değildir. İlk commit03c3dda için backend/frontend/Compose CI başarılı; ML/research son gözlemde çalışıyordu. Bu CI sonucu sonraki commit'ler için geçerli sayılmaz.

## Kullanıcıdan dönen L4 teknik sonuç doğrulandı

Kullanıcı cottonlens-a100-result.zip ve checksum dosyasını gönderdi. Kaynak arşiv hash'i L4teslimatı4c6a206c… ile eşleşti; complete.json record_id, tüm3üye ve2nativeKeras dosyasının hash'leri doğrulandı. Probe bildirimi:NVIDIA L4/23034MiB,TensorFlow2.20.0,NumPy2.1.3,12.702884s,3sentetik train_on_batch/0piyasa fit.

research/verify_gpu_probe.py TensorFlow ithal etmeden nativeHDF5ağırlık/Adam durumunu okudu ve ikinci adımı NumPy ile bağımsız hesapladı. Maksimum mutlak fark1.1920928955078125e-7; ön kayıt toleransı1e-6. Rapor research/evidence/l4-technical-verification-20261106.json. Bu donanımın uzaktan kriptografik attestation'ı, piyasa öğrenme gücü veya G1/G2kabulü değildir. Oturumun zorla kesilmesi/çokworker/arşivrestore dahil bütünG3maddeleri tamamlanmış sayılmaz.

Verilen iki dosya output/data-gpu-audit/l4-returned altında kopyalandı; kullanıcı Downloads orijinalleri değişmedi. Girdi/üye bozulması için2arşiv-yapısı testi ve Ruff geçti. Eski CI'daki11sentetik fixture eksikliği6test modülünde giderildi; bu grupta45test geçti. Son kodun tamCI'ı ayrıca beklenir.

## G2 fonksiyonel nedensellik kontrolü

Temporal audit sentry eklendi: future join, centered rolling, global scaling ve backfill enjeksiyonlarını reddeder; kaynak mutasyonu ve origin düşürme de reddedilir.6sentetik test geçti. Son median deneyinin338completed market kaydından169tekil cutoff çıkarıldı; her cutoff'ta tam tarihçe/prefix ve gelecekteki sayısal gözlemleri değiştirerek5fiyat feature'ı byte/float-exact eşleşti. Yeni fit0. Makbuz research/evidence/price-feature-prefix-audit-20261106.json; tekrar üretim CLI research/recheck_price_features.py. Kontrol kayıtlı satır nedenselliğiyle sınırlı; yayın/vintage kanıtı, bütün diğer feature builder'lar veya bütün seçim politikaları için geçerli değildir.

884c35c commit'inin PR CI42189277401 çalışması success oldu: backend/frontend/ML-research/Compose. Bu başarı sonraki temporal-audit commit'ini kapsamıyor. G1 veri boşluğu devam ediyor; yeni market kampanyası açılmadı.

## Doğrudan QLIKE optimizasyon açığı kapatıldı

Üst adapter kabul kapısına rağmen doğrudan `qlike_har` çağrısı optimizasyon
başlatabiliyordu. Runtime izni ve tam eğitim frame'i/ufuk/sıralı feature kimliği
artık scaler istatistiği, OLS başlangıcı ve BFGS öncesinde zorunlu. Scope dışında
veya scope bittikten sonra çağrı reddedilir. Label/split/ufuk/feature değişikliği
testlerinde `minimize` hiç çağrılmaz; geçerli sentetik örnek fit olur.

Opt-in sentetik backend fixture artık yalnız kontrol fonksiyonunu geçici olarak
yetkilendirmek yerine adapter'ın tüm çağrısı boyunca gerçek admission scope'u
tutar. Kernel sayısal testleri de yalnız üretilmiş veriye açık fixture kullanır.
72 QLIKE/named-CT-offset/statistical/admission testi ve değişen Python dosyalarında
Ruff geçti. Bu sayı bütün suite veya gerçek veri kabulü değildir.

[Tek hata sicili](../research/data-gpu-defects.json), kapatılan kod kusurlarını
kaynak ve doğrulama açıklarından ayırır. Tarihsel ESMIS katalog kapsamı için
[ayrı arşiv makbuzu](../research/evidence/early-catalog-review-20261106.json) eklendi.
41ffe0e commit'inin PR CI42190338412 çalışması dört job ile başarıyla tamamlandı;
bu bölümdeki sonraki kernel değişikliğini henüz kapsamaz.

Tam ML+research çalışması1.624passed/4failed/3skipped verdi; dört başarısızlık
research sentetik fixture scope süresindeydi. Fixture, adapter çağrısının tamamı
boyunca kimlik denetimini tutacak biçimde düzeltildi; üretim kapısı gevşetilmedi.
Bu ilk run bütün suite başarılıymış gibi sunulmaz.

Fixture düzeltmesi sonrası bütün research/tests ve etkilenen ML grubu birlikte
371passed/74.18s verdi. Değişen beş Python dosyasında Ruff ve diff kontrolü geçti.
Tam ML suite fixture-only düzeltmeden sonra yerelde yeniden çalıştırılmadı;
son commit CI sonucu ayrıca gerekli.

## İç seçim sınırı yeniden kontrolü

Engine inner/tune ve full-year inner_price girişleri, hazırlanmış fold'a
körü körüne güvenmek yerine tam3×63origin, sıralama/tekillik/ayrıklık,
kayıtlı kaynak satırlarını sıkıştırmama, cutoff eşleşmesi ve dış origin'den
kesin önce olgunlaşmış T+1/T+5hedeflerini denetler. T+1tarifte de mevcut
T+5purge gevşetilmedi. Bozuk örnekler hiçbir fit'e ulaşmaz; geçerli örnek
hata enjeksiyonundan sonra da geçer. Bu guard'ın önce eksik olması geçmişte
sızıntı meydana geldiğinin kanıtı değildir.

78selection/research/full-year testi ve değişen Python dosyalarında Ruff geçti.
Son median deneyinin hash'e bağlı history/ready/split dosyalarıyla8fold yeniden
kontrol edildi; her yılda3×63iç origin, toplam2006dış origin korunuyor. Yeni fit0.
Makbuz research/evidence/price-selection-boundary-20261106.json; tekrar üretim
CLI research/recheck_selection.py. Bunlar kayıtlı tarih sınırlarıdır; ilk yayın,
vintage, bütün seçim yöntemleri veya bütün tahmin değişmezliği kanıtı değildir.
G1/G2tamamlanmadı.

7a9f862commit push CI42244166480başarılı tamamlandı. Bu başarı sonraki seçim
koruması commit'ini kapsamaz.

İlgili nonlinear-path/transfer testleri8passed/2skipped verdi. İki skip mevcut
isteğe bağlı bağımlılık koşullarıdır; bu çalışmada gerçek GPU testi yapılmadı.
Validator'ın ham byte snapshot'ı, CLI ve makbuz ayrı
selection-boundary-audit-20261106.zipasset'inde saklandı; taze indirmede bütün
üyeler ve makbuzdaki validator hash'i eşleşti. Teslim makbuzu:
research/evidence/selection-boundary-delivery-20261106.json. Bu son kontrol
pipeline çalıştırması değil, arşiv üyelerinin byte/kimlik kontrolüdür.

## Ledger sahipliği ve ani kapanma

Doğrudan Ledger.run çağrısı artık cache/fit/upload/summary güncellemesi boyunca
ledger/execution/.writer-lock altında tek yazıcıdır. Farklı worker'lar aynı
ledger'a yazamaz; ayrı ledger/result alanı gerekir. Eski campaign-level lock
korunur. Hard-kill sonrası lock süreye veya PID tahminine bakılarak çalınmaz.

Fit girişimi başlamadan run_id/spec bağlı started kaydı, doğrulanmış staging
sonrası computed kaydı yazılır. Yakalanan hatalar run_id ile ilişkilidir.
Başlamış ama computed/failure kanıtı olmayan girişim unresolved kalır; bitmiş
fit sayılmaz. Tam optimizer devamı iddia edilmez. Metadata hash'i ve run_id
uyuşmazlığı summary'de reddedilir. Verified staged restore computed olayını
yayımlayabilir; bu işlem yeniden eğitim değildir.

Beş fault testi gerçek ayrı süreç rekabeti ve os._exit ile ani kapanma, enjekte
OOM-benzeri hata, bozuk staged payload ve değiştirilmiş journal kontrolünü geçti.
104 ilgili ledger/admission/research/information testi önce geçti; son iki yeni
fault testi dahil5fault kontrolü ayrı JUnit makbuzuyla geçti. Ruff geçti.
İlk hard-exit testi Windows venv launcher PID'sini interpreter PID'si sandığı
için başarısızdı; test gerçek owner PID'sini kaydedip Windows process durumunu
salt-okunur sorgulayacak şekilde düzeltildi. Üretimde otomatik lock silme yok.

Kanıt research/evidence/ledger-fault-controls-20261106.json. Bunlar üretilmiş
byte fixture'larıdır: yeni piyasa/native GPU fit0. GPU OOM, optimizer/random
state ve Colab oturum kopması bütünüyle doğrulanmış değildir; G3kısmi kalır.
6c5fd9f commit CI42245469238 başarılı; yeni ledger değişikliği ayrıca CI bekler.

JUnit sonucu research/evidence/ledger-fault-tests-20261106.xml olarak saklandi;
yalniz gereksiz yerel hostname cikarildi, test adi/sonuc/sureler korundu.
Makbuz hem orijinal yerel hem disa aktarilan XML hash degerini belirtir.

## User-run L4 native worker restart handoff

ml/gpu_restart_probe.py runs a bounded generated-data test in separate Python
processes. The baseline exits cleanly before the crash worker acquires the GPU.
That worker saves the native checkpoint and optimizer, then exits via os._exit(73).
A fresh process resumes one Adam step and compares all weights, optimizer variables
and predictions to the uninterrupted reference at absolute tolerance1e-6.
Total4train_on_batch calls;0market fits. Completed cache is hash-checked; partial
or corrupted output stops without automatic retraining. No stochastic RNG, real
GPU OOM or whole Colab-session restart acceptance is claimed.

11local refusal/orchestration tests and Ruff passed. Notebook cells parse; all144
source members match their hashes. New distinct files were uploaded to the existing
Drive folder: CottonLens_L4_Restart_Drive.ipynb, cottonlens-gpu-restart-source.zip,
restart-source-manifest.json. The UI showed3completed uploads; old files remain.
The notebook reads its ZIP from Drive without manual upload. User runs a fresh
L4runtime and returns cottonlens-gpu-restart-result.zip and its.sha256.json.
No Colab browser execution was performed by the assistant.

A separate private Release handoff asset was freshly downloaded and all3members
matched byte-for-byte. Receipt: research/evidence/gpu-restart-handoff-20261106.json.
Drive itself was not freshly downloaded; runtime source hashes remain mandatory.
Actual GPU result is pending.5d5b285PR CI42247008149 succeeded; this follow-up needs
its own CI. G1 remains closed and G2/G3 remain incomplete.
