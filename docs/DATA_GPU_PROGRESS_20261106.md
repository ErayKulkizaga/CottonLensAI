# Veri/GPU uygulama takip kaydı

Yetkili kapsam: [onaylanan plan](DATA_GPU_EXECUTION_PLAN_20261106.md).
Bu takip kaydı tamamlanma iddiası değildir. Önceki bilimsel kayıtlar değişmez.

| Teslimat | Durum | Mevcut kanıt / kalan iş |
|---|---|---|
| A: Hata ve kapsam sicili | DEVAM | [Hata sicili](../research/data-gpu-defects.json);239 deney kaydı,139/106/28 ve217scored lifecycle sorunu yeniden doğrulandı; bütün gereksinimlerin kapanışı henüz kanıtlanmadı |
| B: Veri sözleşmesi | DEVAM | Role-aware admission ve 99.523 potansiyel hücre ve 58 kontrat için matrisi mevcut; tarihsel listeleme/vintage kanıtı ve kaynak kabulü eksik |
| C: Onarım ve etki yayılımı | DEVAM |139/106/28yeniden doğrulandı; katı kapsam0hazır, tam kaynak onarımı eksik |
| D: Zorunlu eğitim kapısı | DEVAM | Ledger/model/backend/legacy fit ve doğrudan QLIKE kernel kimlik kontrolü; bütün seçim/preprocessing yollarının kapsam denetimi açık |
| D: Sızıntı testi | DEVAM | Kabul/hata enjeksiyonları; beş fiyat özelliğinde 169 cutoff ve 338 sabit native fiyat fit'inde prefix/future testi geçti. Seçim politikaları ve diğer model yollarının kapsamı açık |
| E: Bağımsız değerlendirme | KISMİ | Price-only T+1 exact frozen cohort, Decimal MAE/yön/yıl/aktif oran ve native replay doğrulandı; diğer sözleşmelerin kapsamı açık |
| F: GPU yürütme/restore | KISMİ | Kullanıcının onayladığı L4 sentetik/native Adam replay ve uzak arşiv restore geçti; native worker kesinti/devam doğrulandı; tam oturum/RNG/GPU OOM eksik |
| G: Kampanya manifesti/sonuç | G1/G2/G3 BEKLİYOR | Veri kabulü olmadan yeni piyasa fit'i yok |
| GitHub teslimatı | DEVAM | Draft PR77 ve checksum bağlı private Release mevcut; 3741386 başlığında PR/push CI başarılı; sonraki teslimat kendi CI sonucunu bekler |
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

## User-returned L4 worker restart verified

The returned result ZIP SHA256 is bc16b482fd0b2ad958496e585730798788119ab4623cd993a1b487616203ec42.
All8members, four self-hashed JSON records, source archive81d0d1f... and both
executed script hashes match the delivered handoff. Hardware is reported as
NVIDIA L4/23034MiB; baseline and resume report TensorFlow2.20.0. Four generated
train_on_batch calls,0market fits. The worker exit report is73.

Without importing TensorFlow or deserializing executable Keras objects, the
verifier reads finite native HDF5 arrays with parent-link checks. Interrupted
weights/Adam state match step1exactly; resumed weights/Adam state match step2
exactly (both max absolute difference0). Baseline Adam step is independently
recomputed in NumPy: max error1.1920928955078125e-7 <1e-6. Keras shared_object_id
is ignored only as an in-memory serialization alias; recipe fields still match.

Five archive/record/config/external-link tests and Ruff pass. Original Downloads
files remain intact; copies and the initial verifier receipt remain in output.
The private Release received l4-native-restart-returned-20261106.zip. All5members
were freshly downloaded/checked; the fresh payload produced the same independent
verification result. Receipts: research/evidence/l4-restart-verification-20261106.json
and research/evidence/l4-restart-delivery-20261106.json.

This verifies deterministic native worker-exit continuation, not whole Colab
session recovery, stochastic RNG/dropout continuation, real GPU OOM or remote
hardware attestation. G3 remains partial and source G1 remains closed; no new
market campaign is admitted. User need not repeat this fixed successful probe.
d56987dPR CI42248201986 succeeded; subsequent verifier changes need their own CI.

## Lifecycle-first source request matrix

The strict unresolved dependency panel was hash-checked and reduced to58contract
requests without changing any source cell or training gate.40,450upper-bound cells
span766source dates:38,397no accepted table,1,593partial weekly prefix,460identity
or price conflicts. The1,947,792cell-origin dependency edges match independently
when summed from the cell panel and the origin panel; they are not unique fits.

research/source_request_matrix.py emits the per-contract CSV and input/output
hash receipt under research/evidence/source-request-matrix-20261106. Five positive
and injected duplicate/count/interval/unknown-kind tests and Ruff passed. Requests
now explicitly ask for official listing/last-trade metadata before paying for
prices of potentially unlisted contracts. No lifecycle date or absent status was
inferred; no provider contact/purchase or new fit occurred. Conditional annual
source and PIT/vintage gaps remain open. The data request document links the exact
58contracts and date ranges, including2024delivery months used before2024origins.

## Complete report is not complete exchange listing

A primary ICE production expiry page and its CSV download show15current futures
with FTD/LTD fields. These current rows do not cover our58historical contracts;
no old listing date was inferred. The previous readiness path nevertheless used
complete ten-price reports and conditional annual tables as absence coverage.
That implication is removed. quote_cell requires a separate verified absence
coverage flag; no existing report constructor supplies one. The independent
verifier makes the same policy correction with its own literal pool builder.

Real-source zero-fit rerun: potential unknown cells40,450→99,523; source dates766→3108.
2006eval/2824requested origins/174jobs are unchanged.1955target pairs and1647HAR
ready remain; EWMA and all-input readiness remain0. Independent verification agrees.
The request matrix was separately versioned; the earlier matrix and exact original
generator bytes are preserved. New counts include58,759cells outside reports or
annual quotes without listing-coverage evidence. These are uncertainty bounds,
not observed price errors.32focused tests and changed-file Ruff passed.

Private Release asset listing-coverage-review-20261106.zip preserves227manifested
members plus the manifest, including current source snapshot, raw official HTML/CSV,
new readiness outputs and both matrix versions. Fresh download/restoration verified
every member; offline/training-disabled replay regenerated8outputs byte-identically.
Receipt: research/evidence/listing-coverage-delivery-20261106.json. The full research
suite passed310tests with1optional HDF5-dependency skip in the pinned CPU environment;
changed-file Ruff and diff checks passed. No source accepted and no model fit run.
Previous commit59f5a7cPR CI42272556465 succeeded; the listing correction needs own CI.

## Independent evaluation bound to frozen origins and targets

The independent median verifier previously checked 2006unique origins per arm and
cross-arm equality without checking exact frozen fold/history alignment. Both
arms could share a wrong date/target metadata value. It now independently checks
the ready record identity/history hash, exact origin order/coverage, target dates,
prices/returns, year/horizon and finite predictions. No intersection/drop repair
is performed. This missing guard is not proof the actual historical export was
wrong: the authentic export passed all new checks.

Sixteen generated cohort/loss tests passed, including both-arm defects, omitted
predictions, duplicates, order/year/horizon changes, nonfinite predictions, Naive
zero-correction identity, zero realized movement, perfect prediction and unit
scaling. The first scaling test used default28-digit Decimal arithmetic against
50-digit results; it was corrected to test50-digit rounding at1e-43. This does
not change scientific thresholds. Main MAE tolerance remains1e-12 and CI1e-11.
All research tests:326passed/1optional HDF5skip; changed-file Ruff passed.

The original private result Release was freshly downloaded: archive ddf449e4...
and manifest7948cef...matched recorded identities;2085files were restored. Frozen
native verification ran with network and training denied.338existing market plus
4existing synthetic fits and train-only preprocessing replayed; the338market
fit aggregate has0native return difference (controls passed their1e-7tolerance). All2085files still matched their original hashes afterward;0new fits.

The independent calculator rechecked12,036Decimal price losses, raw/selected MAE,
direction, individual annual gains/year wins, active rates and active-only direction,
and both preregistered10000replicate block20/60 primary intervals. Both arms have
the same exact2006frozen T+1origin/target pairs; decision remains STOP. This is
frozen dataset consistency, not vendor PIT or named-contract truth.

Receipts: research/evidence/price-independent-cohort-20261106.json and
research/evidence/price-independent-native-20261106.json. A small supplemental
private Release asset price-cohort-audit-20261106.zip adds the current verifier,
tests and receipts without reuploading original data/models. Its fresh download
and independent output matched byte-for-byte; delivery receipt is
research/evidence/price-cohort-delivery-20261106.json. G1 and full G2/G3 remain open.
Previous cbfa6c3PR CI42274764716 succeeded; current changes require own CI.

## Primary historical-rule lead and pending provider inquiry

The connected TradingView health check returned UNAVAILABLE/HTTP404; no chart,
account, integration settings or alert was changed and no data was retrieved.
No candidate paid-provider credential names were present in the process env;
values/secrets were not read or printed. This is not proof no account exists.

A primary ICE Cotton Rule10.07 PDF provides the default35-month listing schedule,
with a November16,2007effective annotation and Board-override language. The source
PDF was captured and hashed. A conditional, zero-admission schedule diagnostic
matches all15current FTD months and partitions the58historical contracts'99,523
unknown cells:83,312before the normal listing month;1,216within;14,995after.
Current rows do not prove historical dates; exceptions and actual first Business
Days remain unchecked. The main readiness source cells and all G1states are unchanged.

research/ice_listing_schedule.py and8generated tests verify month arithmetic,
invalid delivery months, duplicates and preservation of all unknown cells. Ruff
passed. The private Release asset ice-listing-rule-review-20261106.zip preserves
the raw PDF, current expiry input, code, outputs and inquiry draft. Fresh download
checked9manifested members; offline replay reproduced both outputs byte-identically.
Receipts: research/evidence/ice-listing-schedule-20261106/result.json and
research/evidence/ice-listing-rule-delivery-20261106.json.0new fits,0cells admitted.

The ICE helpdesk request is concrete in docs/ICE_DATA_REQUEST_DRAFT_20261106.md.
It attaches only the58contract metadata/request CSV and asks for actual lifecycle,
exception history, a fixed settlement sample and written licensing/price quotes.
User authorization was requested asynchronously because AGENTS.md requires consent
before external messages; no message was sent and no purchase was made.
Prior70fa21bPR CI42276527781 succeeded. Final-head CI remains separate.

## User-reported ICE send; fixed-native prediction invariance

The user reported sending the approved ICE inquiry. No duplicate message was sent
by the assistant; actual sent content/attachments, delivery and any reply are not
independently verified. research/evidence/ice-inquiry-status-20261106.json records
this distinction. The purchase and source-admission gates remain closed.

The interrupted prediction audit process handle was no longer available, but its
complete result existed. All338unique market records and the pipeline/builder/audit
source hashes were checked before accepting that result; it was not blindly restarted.
Each frozen JSON model/preprocessing and actual outer selection weight was held fixed.
At first/last test origins the full-history, prefix and future-price perturbation
predictions matched exactly. Inner fits were checked as raw validation predictions;
outer selected predictions also matched their recorded annual outputs within1e-7.
No scaler/target/model fit or model selection ran in this audit.

The row-wise audit reduction differs from saved native predictions by at most
3.6410563938882312e-9, below the pre-existing1e-7native-equivalence tolerance. Exact
causal invariance is tested within the same fixed arithmetic path, not confused
with differing BLAS batch roundoff. Model/recipe/selection mutation and generated
future-scaler/future-selection injections are rejected.10temporal controls and
changed-file Ruff passed. The first end-to-end receipt is
research/evidence/price-prediction-causality-20261106.json.

This is fixed price-only Ridge/Quantile inference causality for this338fit corpus,
not universal selection-policy validity, vendor availability or all model-family
acceptance. Native/source/readiness gates and STOP decision are unchanged.

Fresh private Release download checked142manifested members of
price-prediction-causality-20261106.zip. Restored source reproduced the full
338-record prediction report byte-identically with network and training disabled;
0new fits. Receipt: research/evidence/prediction-causality-delivery-20261106.json.

Final local checks for this delta:334research tests passed,1optional HDF5 skip;
10temporal ML controls passed; changed-file Ruff and staged diff check passed.
Pinned CPU environment lacks Ruff, so Ruff ran with the existing host installation.

## Positive selection-loop controls

All three actual selection loops (Experiment.inner, full_year.inner_price and
Experiment.tune) now complete on generated valid data with a deterministic fit
callback. The callback checks delivered training maturity and excludes outer
origins. Changing outer-period prices/returns leaves results and execution traces
identical.18selection-boundary tests passed. These callbacks fit no models and
do not certify backend early stopping or all selection policies.


## Güncellenmiş planın bağımsız işleri

Aktif uygulama worktree'si naive-mae-20261105/CottonLensAI, dal
codex/root-cause-audit-20261106. Ana eski checkout'taki kullanıcı değişiklikleri
korundu. PR77 taslak/açık;3741386için42283410170ve42283397748başarılı.
Kullanıcının güncel plan değişikliği ve yerel Cotcast inceleme notu korunuyor.
Plan yeni genel planla değiştirilmedi. ICE yönlendirme bilgisi kullanıcı planı
beyanıdır; data@ice.com'a mesaj gönderilmedi, veri yanıtı kabul edilmedi.

Yeni çıktılar docs/DEMO_WEATHER_ACCEPTANCE_20261106.md içinde:
- Değişmez manuel demo kayıt aracı ve9hata/pozitif kontrol; kimlik/gerçekleşme
  kabulü olmadığı için birincil değerlendirme kapalı. Mevcut tarayıcı sekme
  listesi boş; bu tur yeni canlı gözlem oluşturulmadı.
- Mevsimsel hava kaynak kararı DEFER_SOURCE_ADMISSION. Sayısal NASA kataloğu
  bulundu, retrospektif ürün ile tarihsel yayın ayrımı korundu. Yeni ön kayıt
  veya piyasa eğitimi yok.
- Ayrı ürün ekranı gereksinimleri; backend/frontend/API değişmedi.
- L4 bounded allocator-OOM/fresh-process recovery notebook/kaynak paketi hazır.
 145kaynak üyesi ve notebook AST/UTF8 doğrulandı; private Release'ten taze
 indirilip3dış/145iç dosya hash'i eşleşti. Gerçek GPU çalışması henüz yapılmadı;
 bu yüzden DG007/G3 kapanmadı. Önceki başarılı restart çalışması tekrarlanmaz.

Yerel doğrulama:343research testi geçti,1opsiyonel HDF5 atlandı; yeni OOM
kontrol akışı ve demo kontrolleri13test geçti; değiştirilmiş Python dosyalarında
Ruff geçti. Bu testlerde gerçek model fit'i veya TensorFlow/CUDA kurulumu yok.

Kalan bağımlılıklar: kaynak G1 için gerçek lifecycle/PIT ve lisans kanıtı;
Colab için kullanıcının yeni OOM paketini çalıştırması; session/RNG resume ve
kalan G2 kapsam denetimi. Kampanya manifesti veri kabulünden sonra kilitlenecek.

Ek sınır kontrolü: gözlem saati UTC olarak normalize ediliyor; aynı anın+03:00
veZgösterimleri yeni revizyon üretmiyor, çağıranın girdisi değişmiyor.
10demo kontrolü ve Ruff geçti; önceki343research sonucu bu ek testten öncedir.

## Returned L4 bounded OOM evidence — 7 November 2026

User-returned result ZIP SHA f97e452329db25172ccc44d90721eaa664915e75e4b401c5590bd4c7d5306251
matches its checksum and the delivered source archive336f5bea. All three record
identities and file closure pass offline semantic verification without TensorFlow.
Reported L4/TF2.20.0 caught ResourceExhaustedError for256MiB under128MiB logical
limit; fresh-process GPU recovery returned[1,2] and binds the exact OOM receipt.
No new fits. This is returned receipt verification, not remote hardware attestation
or physical VRAM saturation. Whole-session and stochastic resume remain open.
Eight generated controls pass, including rehashed semantic corruption injections;
Ruff passes. See research/evidence/l4-oom-verification-20261107.json.

User preference is persisted in AGENTS.md: batch compatible remaining checks in
one resumable notebook, deliver directly to Drive, preserve prior successful
results. No separate new notebook is generated in this review.

Private Release fresh download checked5manifested files and reproduced the
OOM verification exactly without TensorFlow/new fits. Receipt:
research/evidence/l4-oom-result-delivery-20261107.json.

## Consolidated remaining session/RNG handoff — 7 November2026

One two-runtime Drive notebook now carries the three prior verified receipts
without rerunning old probes. Explicit NumPy PCG64/TF Philox batch/mask state,
native model/Adam and prediction continuation are prepared. New boot identity
and absent local marker are required; incomplete attempts cannot silently retry.
Twelve focused controls and Ruff pass. An earlier broader run had357passed/1skip
including4session phase controls, before the7new verifier controls were added.
Actual L4 execution is pending; no local TensorFlow, no market fits.
See docs/L4_REMAINING_CHECKS_20261107.md and the handoff receipt.
Drive UI confirmed2uploads; remote download hashes were not claimed.
