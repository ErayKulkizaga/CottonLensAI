# Deney geçmişi ve tekrar kontrolü

## Sicil

`curve-carry-attribution-v1`: **sıfır-fit karşıolgusal mekanizma doğrulandı**.
T+5'in749 origin/2996 kayıtlı tahmini korunur;743 kaynak çifti,6 bilinmeyen.
Ham D0'ın74 geçiş origin'indeki +24,657265 katkısı gap çıkarılınca
−3,367436;669 aynı-kontrat katkısı değişmez.15 olay bağımsız74 olay
değildir;karşıolgu yeni hedefte OOS beceri ölçmez. Kaynak performansı
INCONCLUSIVE,özgün karar değişmez. [Tanı ve tek sonraki sıfır-fit adım](../docs/CURVE_CARRY_ATTRIBUTION_20261010.md).

`contract-curve-t5-pilot-v1`: **tamamlandı,kaynak katkısı INCONCLUSIVE**.
252 piyasa +1 ayrı sentetik fit;749 origin/kol,12 çıktı/2.996 satır.
Seçilmiş D0 Naive kazancı −%0,549260,ham −%7,242578;253 çıkarım/12 seçim/
16 aralık ve exact rapor/CSV/geçiş açıklaması doğrulandı. Saat/ilk vintage
varsayımlı,üç yıl6/8 koşulunu değerlendirmez.74 geçiş origin'inde ham
kaynak katkısı pozitif,669 aynı-kontrat origin'inde negatif;gelecek kontrat
hiçbir feature/seçime girmedi. [Sonuç ve tek sıfır-fit sonraki adım](../docs/CONTRACT_CURVE_T5_RESULT_20261010.md).
Aşağıdaki sıfır-fit ön kayıt kaydı bu sonucun tarihsel önceki aşamasıdır.

`contract-curve-t5-pilot-v1`: **sıfır-fit ön kayıt, piyasa sonucu yok**.
Tamamlanmış T+1 history/özellik/749 origin'i korunur; yalnız ufuk5.252
piyasa +en fazla1 ayrı T+5 sentetik fit tavanı; yeni checkpoint namespace'i.
Gelecek kontrat metadatası yalnız sonuç açıklamasında, seçimde değil.
Saat/vintage varsayımlı;3 yıl6/8 gate'ini değerlendirmez. [Kayıt ve tek sonraki yürütme](../docs/CONTRACT_CURVE_T5_PROTOCOL_20261010.md).

`curve-target-integrity-v1`: **sıfır-fit, sonuç-sonrası hedef/proxy denetimi**.
749 origin/ufuk korunur;743 quote çifti/ufuk,6 bilinmeyen. T+5'te74 origin,
15 gözlenen ilk-vade değişimi. Gelecek kontratını sonradan bilen gap-only
azalma %2,891947 **OOS/model becerisi değildir**; tanı satırları tahmin değildir.
1.486 Decimal ayrıştırması/1.498 hedef tarihi/5 ret kontrolü; kaynak/veri
değişmez. Tahmin katkısı sınıfı INCONCLUSIVE; T+5 ön kayıt incelemesi henüz
eğitim izni değil. [Sonuç ve tek sonraki karar](../docs/CURVE_TARGET_INTEGRITY_20261010.md).

`contract-curve-t1-pilot-v1`: **tamamlandı, kaynak katkısı INCONCLUSIVE**;
252 piyasa +1 ayrı sentetik fit, 749 ortak origin/kol, 12/12 ağırlık0.
Ham numeric D0/D1 Naive kazancı −%2,711585/−%2,282736; seçilmiş `[0,0]`
kaynakta bilgi yokluğu değildir. 253 çıkarım/12 seçim/16 aralık ve exact
rapor/CSV replay doğrulandı. Saat/vintage varsayımı, geçmiş yılların reuse'u
ve ölçülemeyen 6/8 gate'i korunur. [Sonuç ve tek sonraki ön kayıt incelemesi](../docs/CONTRACT_CURVE_RESULT_20261010.md).
Ayrı sıfır-fit ön kayıt kaydı tarihseldir; bu sonuç eski kaydı silmez.

`existing-information-admission-v1`: **sıfır-fit kaynak/kapsam elemesi**;
1.000 eski girdi, 997 tablo /9.970 gerçek vade fiyatı. Eski spot-basis
T+5 bunu sınamadı; 2016–2019 yok, UTC/ilk-vintage kabulü yok. Kaynak
model-eligible 0; INCONCLUSIVE tahmin becerisi sorusuna ilişkindir, fiyat
envanterinin doğrulanmadığı anlamına gelmez. Yeni model tarifi/fit yok.
[Kabul matrisi ve sonraki tek karar](../docs/INFORMATION_ADMISSION_20261010.md).

**Aşağıdaki ön kayıt aşaması tarihsel bilgidir; tamamlanmış sonuç kaydı bu
sayfada aşağıdadır ve [sonuç raporuna](../docs/PAIRED_PRICE_LOSS_RESULT_20261010.md) bağlıdır.**

`paired-price-loss-control-v1`: **sıfır-fit ön kayıt; piyasa sonucu yok**.
48 sabit CPU XGBoost fit'i / 2.688 ağaç ve ayrı 6 öğrenme kontrolü;
iki price_delta kolunda yalnız MSE/MAE değişir, 1.008 ortak origin.
Eski MAE grid'ini tekrarlamaz; tarihsel parametre/iteration'lara koşullu
loss kontrolüdür. [Protokol](../docs/PAIRED_PRICE_LOSS_PROTOCOL_20261010.md).

`legacy-loss-lineage-v1`: **sıfır-fit eski seçim/kayıp yeniden kurması**;
6.648 makbuz, 128 aday × 8 dönem, 1.008 ortak T+1 origin. Seçilmiş eski
fiyat programı −%0,6110, 1/8 dönem; doğrudan fiyat-MAE loss'u denenmişti.
Kontrollü loss ablation'ı INCONCLUSIVE; model payload'ları yok. Generic
MAE grid'ini tekrar etmeyin. [Kapsam ve tek küçük kontrol](../docs/LEGACY_LOSS_LINEAGE_RESULT_20261010.md).

`price-mae-calibration-v1`: **tamamlanan piyasa postprocessing deneyi**;
0 yeni model fit'i, 169 skaler kalibrasyon, aynı 2.006 T+1 origin.
Ham/seçilmiş Naive kazancı −%3,73685/−%0,02608; kontrol katkısı belirsiz.
Bu global çarpan tarifi için DECISIVE_NEGATIVE; tam koşullu quantile modeli
ve tüm kaynaklarda bilgi yokluğu sınanmadı. Model fit sicili değişmez;
skaler tahminler ayrı envanterlenir. [Sonuç ve kapsam](../docs/PRICE_MAE_CALIBRATION_RESULT_20261010.md).

`weak-signal-selection-stability-v1`: **tamamlanan sıfır-fit tanı**;
160 geçmiş seçim / 1.440 iç makbuz / 30.240 doğrulama tahmini.
Blok20 enjekte koşulunda kayıtlı ağırlığı medyan %56,19 sıklıkla seçer;
leave-one-block-out 59/80 kararı değiştirir. Blok60 küçük başlangıç kümesi
nedeniyle güç kanıtı değildir. Eski ağırlıklar ve fit sicili değişmedi.
[Sonuç ve sınır](../docs/SELECTION_STABILITY_RESULT_20261010.md).

`weak-signal-control-v1`: **tamamlanan sentetik duyarlılık kontrolü**;
10 seed × null/enjekte, 3.380 sentetik fit, 160 çıktı, 40.120 satır; piyasa fit'i 0.
Ham/seçilmiş ortalama kazanç %1,9551/%1,8752, oracle %3,6054. Ön kayıtlı
RAW_ONLY_RECOVERS kararı sınırdadır: seçilmiş medyan koruma %49,5611 ve 10/10
seed pozitiftir. Bu piyasa başarı/negatif kanıtı veya küçültmenin ana hata
olduğunun kanıtı değildir. [Sonuç ve tek sonraki analiz](../docs/WEAK_SIGNAL_RESULT_20261009.md).
[Ön kayıt](../docs/WEAK_SIGNAL_PREREGISTRATION_20261009.md) değişmedi;
tamamlanmış 20 sentetik tarif piyasa tariflerinden ayrı indekslenir.

İlk sıfır-fit ön kayıt Linux test fixture yolunda başarısız oldu; hiçbir fit
başlamadı. Aynı 20 sentetik geçmiş ayrı `research-weak-signal-control-v1-r2`
kimliğiyle yürütüldü. İlk kayıt negatife çevrilmez ve arşivi değiştirilmez.

`nass-regional-t1-pilot-v1`: **tamamlanan sınırlı piyasa deneyi**, 676 piyasa
fit'i + ayrı 1 sentetik kontrol; dört kol × 2.006 ortak origin. Seçilmiş sayısal
D0/D1 Naive kazancı −%0,0274/−%0,2556; iki gecikme/blokta %5 üst sınırın dışında.
Bu sabit tarif için DECISIVE_NEGATIVE; bütün Texas bilgisinde sinyal yokluğu
veya doğrulanmış vintage/saat sonucu değildir. [Sonuç/yeniden çıkarım](../docs/NASS_REGIONAL_T1_RESULT_20261009.md).
Eski sıfır-fit [ön kayıt](../docs/NASS_REGIONAL_T1_PREREGISTRATION_20261009.md)
değişmedi; tamamlanmış piyasa ve sentetik kayıtları ayrı tutulur.

`nass-clock-case-v1`: **sıfır-fit tek raporlu saat incelemesi**; 22 hücre
eşleşir, tarihsel sürüm/erişim kanıtlanmadı. Üç CDX timeout/503, tekrar sorgu yok;
bu bir piyasa negatifi değildir. [Kanıt ve çevrimdışı replay](../docs/NASS_CLOCK_CASE_20261009.md).

`nass-texas-report-audit-v1`: **sıfır-fit bölgesel içerik denetimi**; 311 rapor,
636 gelişim tablosu, 949 girdi. Ulusal T+5 deneyi Texas'ı elemez. İki gerçek
revizyon ayrı korunur; tam sezon ve tarihsel erişim kabulü yoktur.
[Kanıt, komut ve devam sınırı](../docs/NASS_REGIONAL_AUDIT_20261009.md).

`paired-price-loss-control-v1` tamamlandı: 48 piyasa + ayrı 6 sentetik fit,
1.008 T+1 origin; 54 native çıkarım/6 aralık/exact rapor replay'i doğrulandı.
MAE–MSE katkısı %1,458; MAE–Naive −%0,4883, yön %48,41, dönem 3/8.
Birincil loss katkısı pozitif; pratik Naive hedefi negatif. Ön kayıt korunur,
tamamlanmış makbuzlar ayrı market/synthetic türleriyle ek indekslenir.
[Sonuç ve arşiv doğrulama komutu](../docs/PAIRED_PRICE_LOSS_RESULT_20261010.md).

`forward-window-audit-v1`: **sıfır-fit, salt okunur gece denetimi**;
00:05/00:20 UTC makbuzları yok, yayın 0; OS uyku aralığı pencereyi kapsıyor.
Daha sonraki 07:05 görev isteği `0x800710E0`; kesin ret nedeni belirsiz.
Altı missing/126-origin kilidi korunur; görev/otomasyon ayarları değişmez.
[Yeni kanıt ve sınırlar](../docs/FORWARD_WINDOW_AUDIT_20261010.md).

`forward-runtime-window-v1`: **sıfır-fit görev düzeltmesi ve gündüz gerçek
çağrı doğrulaması**; ileri piyasa testi değildir. Kaynak kimliği sabit,
yayın penceresi ağdan ayrıdır; altı missing kayıt korunur ve yeni yayın 0.
[Gerçek takvim kontrolü ve sınırlar](../docs/FORWARD_RUNTIME_WINDOW_20261009.md).

`forward-evidence-audit-v1`: **sıfır-fit operasyon denetimi**; altı missing origin,
sıfır yayımlanmış tahmin. Piyasa performans testi değil. Salt okunur komut:
`PYTHONPATH=ml/src python -m cottonlens_ml.research.live --store EXISTING_STORE --status`.
Windows'ta `PYTHONPATH` ortam değişkenini ayrı ayarlayın; mevcut ML CPU ortamı
gerekir. [Makbuz/saat kanıtı ve kalan tek kontrol](../docs/FORWARD_EVIDENCE_AUDIT_20261009.md).

`fas-reviewed-alignment-v1`: **sıfır-fit sentetik sözleşme/hizalama kontrolü**;
FAS specific-version receipt, 00:15 karar sınırı, ortak eksiklik göstergeleri ve
provenance. Gerçek kaynak kabulü veya ülke modelinin piyasa testi değildir.
[Uyumluluk ve kalan koşullar](../docs/FAS_REVIEWED_ALIGNMENT_20261009.md).

`fas-country-quarantine-v1`: 752 hafta × dört kod = 3.008 satırlık **sıfır-fit
hazırlık**, piyasa testi değil. Negatif satışlar korunur; negatif stok bileşeni
olan haftanın stok payları bilinmeyendir. Tarihsel saat/sürüm kabulü kapalı.
[Hazırlık ve eğitime kalan iş](../docs/FAS_COUNTRY_PREPARATION_20261008.md).

Güncel ekler: tamamlanmış bölgesel WASDE T+1 piyasa sonucu ve ayrı sentetik kontrol;
ardından `fas-country-field-audit-v1` **sıfır-fit veri denetimi**. Sonuncusu
ülke dağılımının piyasa katkısını veya tarihsel erişimini test etmiş sayılmaz.
15 kaynak/27.887 ham satır, eski 752 haftalık tabloyla eşleşir; ülke isimleri
ve ülke bazlı rapor değerleri henüz kabul edilmiş değildir.
[Devam sınırı](../docs/RESEARCH_DATA_REENTRY_PLAN_20261005.md).

- `registry.json`: deney/grup/ufuk bazında mevcut OOS kanıtı, origin sayısı, ham/seçilmiş MAE, kaynak/veri kimliği, yıllık kapsam, tahmin dosyalarının SHA-256 değerleri ve kanıt sınıfı. Eski özetler ayrıca **testimony** olarak işaretlidir.
- `trials.json`: gerçekten tamamlanmış, checksum ve dondurulmuş kimliği doğrulanmış ledger **fit makbuzlarının** benzersiz tarifleri. Tek fit ≠ tamamlanmış OOS deney. Model payload'larının yeniden hash'lenmesi bu indeksleme adımının iddiası değildir.
- `evidence/`: legacy LSTM, TCN, zamanlama deneyi ve bağımsız metrik doğrulaması; eski dosyaları değiştiren düzeltmeler değildir.
- `legacy-history.json`: önceki araştırma özetinin değiştirilmeden korunmuş kopyası.
- `organization.json`: taşınan belgeler/notebook'lar, eski yollar, orijinal hash'ler ve nedenleri.
- `archive-summary.json`: Release varlıkları, arşiv manifesti checksum'ı, kapsam ve açıkça dışlanan/gizli bilgileri ayıklanan dosyalar.

Önceki kayıtlar `clock-t1-open-close-proxy-v1` (2.006 origin, sıfır yeni fit,
gerçek işlem becerisi için inconclusive) ve `wasde-regional-candidate-v1`
(95 rapor/26 alan, eğitime kabul edilmemiş veri hazırlığı). İlki yeni model eğitimi,
ikincisi tamamlanmış piyasa deneyi değildir. [Sonuç ve yeni Release](../docs/TRADING_WASDE_DELIVERY_20261005.md).

`INCONCLUSIVE`, bir tarifin hiç denenmediği anlamına gelmez. Tamamlanmamış çıktı, sentetik kontrol veya sıfıra küçültülmüş tahmin tüm kaynakta sinyal yokluğunu göstermez. Sicil evrende denenmiş bütün fikirleri kapsamaz; yalnız mevcut kanıtı kapsar.

## Proje komutları

Depo kökünde, Python 3.12+; ek paket kurulmaz:

```bash
python ml/history.py validate
python ml/history.py check --family xgboost --horizon 5
python ml/history.py check --feature nass --horizon 5
python ml/history.py check --query availability
python ml/history.py check --query price-mae-calibration --horizon 1
python ml/history.py check --query legacy-loss-lineage --horizon 1
python ml/history.py check --query paired-price-loss --horizon 1
python ml/history.py check --query trading --horizon 1
python ml/history.py check --query wasde-regional
python ml/history.py check --query fas-country
python ml/history.py check --query forward-evidence
python ml/history.py check --query forward-runtime
python ml/history.py check --query nass-texas
python ml/history.py check --query nass-clock-case
python ml/history.py check --query nass-regional --horizon 1
python ml/history.py check --query fas-reviewed-alignment
python ml/history.py check --query fas-historical-reference
python ml/history.py check --feature fas --horizon 1
python ml/history.py check --feature fas --horizon 5
python ml/history.py check --query wasde-regional-numeric-verification
python ml/history.py check --query wasde-as-reported
python ml/history.py check --query wasde-clock-case
python ml/history.py check --query semantics
python ml/history.py check --query reconciliation
python ml/history.py check --profile availability-clock-pilot-v1 --json
python ml/history.py list
```

FAS alan denetimi, yalnız mevcut pinned girdileri okur; stdout JSON üretir,
veri indirmez, kaynak/feature/ledger değiştirmez ve eğitim başlatmaz. Girdiler
eski Release'ten ayrı dizine restore edilebilir; `FAS_RAW_ROOT`, iki seviyeli
`request_hash/source_sha/source.json` düzenindeki FAS köküdür. `MANIFEST` ve
`TABLE`, eski `weekly-sales-2010-2023.manifest.json` / `.csv` çiftidir:

```bash
python ml/review_fas_countries.py --raw-root FAS_RAW_ROOT --manifest MANIFEST --table TABLE
```

Aynı pinned girdilerden karantina ülke paneli (stdout JSON, Python 3.12+, paket
kurulmaz; önce/sonra kaynak doğrulaması, başarısızlıkta exit 2):

```bash
python -S ml/prepare_fas_countries.py --raw-root FAS_RAW_ROOT --manifest MANIFEST --table TABLE
```

`week_ending` yayın tarihi değildir; çıktı saat politikası `UNSET` ve
`model_eligible=false` taşır. Doğrudan eğitim/günlük seans hizalaması yapılmaz.

Checksum/şema/ülke-hafta/toplam uyuşmazlığında exit 2; sıfır/eksik değer
uydurulmaz. Başarılı JSON, publication/vintage veya model eligibility onayı değildir.

`fas-country-report-review-v1`: sıfır-fit ülke/rapor içerik kontrolü; 22/24 stok alanı eşleşti, 28 Mayıs Çin/Pakistan birikimli ihracatı −55/−53 balya farklı. Ulusal toplamın geçmesi ülke alanlarının geçmesi değildir. Haftalık akış veya tarihsel yayın/vintage onayı yok. [Sonuç ve tek sonraki kontrol](../docs/FAS_COUNTRY_REPORT_REVIEW_20261008.md). Release girdileriyle `ml/review_fas_archive.py --country-reference` çalışır; sayısal uyuşmazlık kaydı korunur ve exit 2 döner. Bu bir piyasa negatif sonucu değildir.

`fas-may28-subtype-precision-v1`: sıfır-fit açıklama kontrolü. İki fark alt sınıf yuvarlama varsayımıyla uyumlu; bu, tolerans/kaynak kabulünü değiştirmez veya gerçek nedeni kanıtlamaz. `ml/review_fas_archive.py --subtype-precision` mevcut incelemeye tanı ekler; pinned `--country-reference` zorunludur ve doğrudan fark exit 2 kalır. [Tek eksik as-issued hücre ve erişim sınırı](../docs/FAS_MAY28_VERSION_AUDIT_20261008.md).

Önceki iki kayıt `cotton-ohlc-semantics-audit-v1` ve
`ams-cotton-price-reconciliation-v1`: sıfır fit kaynak denetimleridir; yeni
piyasa başarı/başarısızlık deneyi değildir. 998 eski AMS belge hash'i,
997 doğrulanmış rapor tarihi, 996 Close eşleşmesi ve 20 kontrat değişimi
korunur. Tarihi okunamayan bir rapor ve eşleşmeyen bir fiyat açıkça ayrıdır.
[Rapor ve Release](../docs/PRICE_SEMANTICS_AUDIT_20261005.md).

`wasde-regional-numeric-verification-v1` de sıfır-fit veri doğrulamasıdır.
95 raporun sayısal mutabakatı, erişilebilirlik/vintage onayı ve T+1/T+5
piyasa katkısı birbirinden ayrılır. Eski aday kaydı değişmez; bu ek kayıt
ve [doğrulama raporu](../docs/WASDE_REGIONAL_VERIFICATION_20261008.md)
hangi kontrolün gerçekten tamamlandığını gösterir.

`wasde-as-reported-verification-v1`: resmî CSV ile 95 rapor/2.470 alanın sıfır-fit
mutabakatı. 950 temel değer eşleşir; 117 bilinmeyen revizyon korunur. CSV'nin
rapor saati, CSV'nin fiilî teslim saati değildir. Tarihsel kabul ve piyasa becerisi
çıkarımı yok. [Kanıt ve erişim saati ayrımı](../docs/WASDE_AS_REPORTED_20261008.md).

`wasde-clock-case-20181211-v1`: bir raporda iki PDF/XML çifti ve üniversite
kopyasının sıfır-fit değer/saat incelemesi. Cotton değerleri aynı; ilk karar
kesiminden önce erişim doğrulanmadı. Tarihsel kabul, yeni eğitim veya kaynakta
sinyal yokluğu kanıtı değildir. [Karar ve sınır](../docs/WASDE_CLOCK_CASE_20261008.md).

Kesin tarif kontrolü: `trials.json` içinden `scope_id` ve `recipes[0]` alınarak `{ "scope_id": "...", "recipe": {...} }` biçiminde bir öneri JSON'u oluşturun:

```bash
python ml/history.py check --proposal output/proposal.json --json
```

`REPEAT_FIT_RECIPE` (exit **3**) aynı tam tarif ve frozen kapsamda tamamlanmış fit makbuzu bulunduğunu gösterir; başarıyı veya tüm deneyin bitmesini göstermez. `RELATED_EVIDENCE` ve `NO_REGISTERED_MATCH` exit **0**; bunlar eğitim izni değildir. Eksik/bozuk sicil veya kanıt checksum'ı exit **2**, işlem durur. Kaynak/veri/politika/dönem farkı otomatik eşdeğer sayılmaz. Kasıtlı tekrarın bilimsel gerekçesini yeni manifestte kaydedin.

Yeni makbuzları read-only kaynaktan indeksleme, mevcut sicilin üzerine yazmadan:

```bash
# PYTHONPATH=ml/src (Windows: ml/src)
python -m cottonlens_ml.research.history_index --source-root output/recovered --output output/new-trials.json
```

İndeks çıktısını mevcut sicille inceleyerek birleştirin ve yeni deneyin prediction kapsamını `registry.json` içine kaydedin. Sadece yeni profil kodunun bulunması “denendi” kaydı yaratmaz.

## Release yedeği

[GitHub evidence-20261005](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005), bilimsel dosyaları içerik hash'iyle tekilleştirir. Manifest her orijinal göreli yolu, orijinal/public SHA-256 değerini ve varlık/iç yolunu korur. Parçalar bağımsız ZIP'tir; büyük veri Git ağacına eklenmez.

Mevcut ham kaynaklar, model/checkpoint'ler, ledger'lar, tahminler, raporlar, kaynak snapshot'ları ve kesintiye uğramış çalıştırmalar korunur. Yazılım ortamı/cache, kişisel ayarlar, UI ekran görüntüleri ve içeriği güvenle yayınlanamayan veritabanları dışlanır; dışlamalar indekste görünür. Geçmişte kaybolmuş LSTM tahminleri oluşturulmaz. Yerel orijinaller silinmez/değişmez. Ayıklanmış dosya eski hash doğrulamasını geçmez ve eski deney cache'i olarak kullanılmamalıdır.

İndirme/restore komutları ana README'de. Restore araçları model/pickle çalıştırmaz. Başarısız checksum, tehlikeli yol veya mevcut hedef dosya üzerine yazma girişimi durur. Restore edilen eski bilimsel snapshot yeni kodla otomatik devam ettirilmez.

Eski `research-v2-tf-placement` için 7.331 fit ve kısmi on-call için 102 fit makbuzunun model payload kopyaları yerelde yok: toplam 22.299 dosya referansı. Makbuzlar/tahmin kanıtı korunur; bunlar yeniden kullanılabilir tam checkpoint gibi gösterilmez. Yeni zamanlama deneyinin 6.157 dosyası checksum doğrulamasıyla geri kurulmuştur. Kaybolan eski dosyalar yeni eğitimle yeniden üretilmedi.

`fas-historical-reference-witness-v1`: iki 2020 arşiv formundaki dört kod için sıfır-fit tanık denetimi; özgün hücre erişimi `BLOCKED_EXTERNAL_EVIDENCE`. Aynı PDF/form/tek boş sorgu yeni deney diye tekrar açılmaz. Kapsam ve çevrimdışı `ml/review_fas_reference.py` komutu için [tarihsel tanık raporu](../docs/FAS_HISTORICAL_REFERENCE_20261008.md).
