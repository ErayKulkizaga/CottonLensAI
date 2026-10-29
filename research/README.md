# Deney geçmişi ve tekrar kontrolü

29 Ekim: [vade eğrisi sıfır-fit denetimi](../docs/AMS_CURVE_RISK_MOMENT_20261029.md)
kapandı; 2.006 slot/1.987 uygun origin, 5.961 kaynak scalar ve iki bağımsız
blok aralığı doğrulandı. Sabit ekran geçmedi; INCONCLUSIVE, kaynakta sinyal
yok veya model başarılı sonucu çıkarılmaz. Sicil205 aşama/545 tarif/30.419
fit kopyasıdır. Sonradan tamamlanan loss/scale/EWMA-relative HAR ve resmî
günlük named kontrolünü eski aşağıdaki ön kayıt ifadelerinden tekrar başlatma.
Araştırma testleri: `python -m pytest -q research/tests`; bu komut hem pytest
fonksiyonlarını hem mevcut unittest sınıflarını toplar.
Aşağıdaki tarihli bölümler araştırma tarihçesidir; güncel sıra docs/STATUS.md'dir.

Son tamamlanan: [named risk history control](../docs/NAMED_RISK_HISTORY_CONTROL_20261012.md).
`python -m cottonlens_ml.research.named_risk_history prepare|verify|pilot|compare`
aynı HAR/Experiment/Ledger yolunda iki named-label kolunun yalnız history
özelliklerini değiştirir.44market/0newsynthetic fit; eski HAR kontrolü native/
learner bytes/env eşliğiyle reverified, genel passed cache değil.
246slot korunur;243forecast/242paired, üç unavailable açık. Gözlenen22pair
calendar22 değildir; train span236/age192 saptandı. Same-profile tam recipe/
scope lookup tamamlanan22fitperarm bulmalı. Yeni grid veya blind rerun yok.

Son eğitim: [named T+1 risk label control](../docs/NAMED_RISK_LABEL_CONTROL_20261012.md).
`python -m cottonlens_ml.research.named_risk_control prepare|verify|pilot|compare`
doğrudan mevcut Experiment/Ledger ve Log-HAR'ı kullanır. `prepare` frozen
evidence/parent-ready, `pilot` hazır source/data/design/budget ve taze
GitHub geri kurmasına bağlı `--authorization` ister. Ortam/source değişirse
cache reddedilir; eski snapshot ile zero-fit `compare` korunur.
44+1 fit kapandı; etiket iyileşmesi EWMA üstünlüğü değildir. Bu deney
tekrar edilmez. Sonraki named-HAR feature sorusu farklı hipotezdir; önce
`ml/history.py check` ve aynı güçlü referanslı kurulabilirlik gerekir.

12 Ekim son sıfır-fit iş: [risk ölçüm kurulabilirliği](../docs/RISK_MEASUREMENT_READINESS_20261012.md).
`python research/risk_readiness.py --inputs INPUTS --contract CONTRACT --output NEW_NAMESPACE`
yalnız dondurulmuş eski forecast/hedef ölçümü yapar.2.006 origin tüm kayıtlarda
korunur; tam tarihli same-contract label, unknown gerekçeleri ve full-ordinal
bootstrap vardır. CT-trained forecast'un named hedefine transport'u named
öğrenme testi değildir. Yeni fit0. Sonraki yalnız Log-HAR etiket kontrolü
kurulabilirliği; aynı risk hedefli baseline ön kaydı olmadan eğitim yoktur.

12 Ekim aktif sıra: [risk kanıt denetimi](../docs/RISK_EVIDENCE_AUDIT_20261012.md).
T+1 risk/EWMA karşılaştırması mevcut `research-full-year-v1-r2` tahminlerinden
raporlanır; yeni fit0. `python ml/history.py check --family har --horizon 1 --json`
ve aynı GARCH sorgusu ilgili tamamlanmış makbuzları buldu; tekrar fit yok.
`python -m cottonlens_ml.research.risk_audit` açık `--experiment-root`,
`--corrected-root`, `--output` ister; fit stage'i yoktur.22-fit CFTC saat
önerisi ertelenmiş/çalışmamış kayıttır. Risk sonucu fiyat kapısını geçirmez.
Aşağıdaki bölümler kronolojik eski kanıttır.

12 Ekim: [CFTC named37 kaynak kontrolü](../docs/NAMED_CFTC_SOURCE_CONTROL_T5_20261012.md)
48fit kapandı/taze GitHub native geri kurması geçti. Ham kaynak katkısı
+1,809270puan ama belirsiz; numeric Naive−%1,124208 ve iki gain-CI üstü<%5.
Kaynakta sinyal yokluğu çıkarılmaz. Canonical scope check artık aynı44market
fit'i REPEAT_FIT_RECIPE bulur; eski181aşama ve527tarif korunur. Yeni grid yok.

12 Ekim: [korumalı CFTC kaynak kontrolü](../docs/NAMED_CFTC_SOURCE_CONTROL_T5_20261012.md)
yeni37alanlı numeric-mask karşılaştırmasıdır; eski CT CFTC lag grid'i veya
132-fit öğretici kontrol tekrar edilmez. Ön kayıt/taze GitHub geri kurması
olmadan fit kapalı; tarihi UTC/vintage varsayımı ve named kohort sabit.

12 Ekim: [CFTC bilgi kapsamı](../docs/NAMED_CFTC_INFORMATION_AUDIT_20261012.md)
eski CT pilotu ile named hedefin farklılığını sıfır-fit doğruladı. Eski
1.855 fit yeniden sayılmaz/çalıştırılmaz; kaynak etkisi named hedefte
sınanmış sayılmaz. Legacy+3gün audit-only sütunları korumayı atlayarak açma.

12 Ekim: [named weak-teacher sonucu](../docs/NAMED_WEAK_SIGNAL_EXECUTION_20261012.md)
132 sentetik fit ile aynı tarifin basit yararlı etkiyi öğrenebildiğini gösterdi.
DECISIVE_POSITIVE yalnız bu öğretici içindir; piyasa skill/güç kanıtı değildir.
Beklenen ve gerçekleşmiş kazanç ayrıdır; seed2703 farkı saklanır.

12 Ekim: [named zayıf-sinyal yürütme ön kaydı](../docs/NAMED_WEAK_SIGNAL_EXECUTION_20261012.md)
ayrı ek aşamadır;132 sentetik fit henüz çalışmadı. Readiness ve eski piyasa
sonuçları yeniden yazılmaz; taze kaynak/ön kayıt geri kurması eğitim kapısıdır.

11 Ekim: [named zayıf-sinyal readiness](../docs/NAMED_WEAK_SIGNAL_READINESS_20261011.md)
yalnız veri/güç-kontrolü kurulabilirliği; yeni profil fit0 ve performans
kanıtı yok. Aynı32 alan/sabit XGB için üç matched gürültü seed'i ve null/
injected tasarımı,132 sentetik-fit tavanıyla hazırlanır. Öğretici oracle
avantajını öğrenci/gerçek piyasa becerisi olarak sicile yazma. Eski Ridge
T+1 zayıf-sinyal ve noiseless XGB kontrollerinin kapsam farkı korunur.

11 Ekim: [named loss teşhisi](../docs/NAMED_LOSS_DIAGNOSIS_20261011.md)
mevcut PR42 tahminleriyle sıfır-fit kapandı. Yeni fit sicili oluşmadı;
`trials.json` değişmez. Standardized MSE'de her iki kolun iç/dış nokta
kazancı negatif; pozitif MSE başarısını MAE'nin gizlediği gösterilmedi.
Küçük etki aralıkları belirsizdir. Bir sonraki aday için önce sabit named
XGBoost zayıf-sinyal güç kontrolünün geçmişte eşdeğeri aranır; eski Ridge
zayıf-sinyal grid'i veya güçlü XGB sentetiği aynı test sayılmaz.

11 Ekim: [named history kaynak kontrolü](../docs/NAMED_HISTORY_SOURCE_CONTROL_T5_20261011.md)
44+4 fit ile kapandı; 48 native makbuz yeni üç tarif nesnesine eklendi,
eski kayıtlar değişmedi. Ham numeric/mask karşılaştırmasında pratik
Naive üstünlüğü yok; işaret belirsiz. Sonuç ve modeli tekrar fit etmek
yerine `named-history-source-result-20261011` Release'ini checksum ile
geri kur. Tek sonraki iş mevcut tahminlerde sıfır-fit MSE–MAE ayrımı.

11 Ekim: [aynı-kontrat geçmişi hazırlığı](../docs/NAMED_HISTORY_READINESS_20261011.md)
tamamlandı, model fit0. Dört geçmiş getiri mevcut origin karar saatine göre
kuruluyor; 3.520 tarih/14.080 endpoint ve22 sabit eğitim işi doğrulandı.
En küçük528 olgun etikette510 tam-feature satırı;3×63 iç ve246 dış origin
korunur. Numeric/mask eksiklik desenleri aynı;91 regresyon testi geçti.
Bu tahmin katkısı değildir. Tek sonraki iş aynı32 alan/sabit XGB ile
numeric–mask kaynak katkısının ayrı ön kaydı; yeni eğitim henüz başlamaz.
Saat/vintage varsayımı ve eski kararlar aşağıda korunur.

11 Ekim: [sıfır-fit hata ayrımı](../docs/NAMED_CAPACITY_CONTROL_T5_20261010.md#11-ekim-yönbüyüklük-hata-mekanizması-sonuç-sonrası-sıfır-fit) doğrulandı.
Ham XGB'nin yön/büyüklük katkısı +12,319829, doğru-yön aşım cezası
14,680944; net−2,361115 cent/lb toplamı. Yalnız aşım hatalarını geleceği
bilerek kusursuz düzeltmek bile +%1,041249 kazanç sağlar; bu bir politika
veya gerçek başarı değildir. Yeni fit/tahmin0; eski sonuç değişmez.
Tek sonraki iş aynı kontrat geçmişinin mevcut28 özellikte ve saklanan
quote'larda temsil/erişilebilirlik denetimi; yeni model/grid başlamaz.

11 Ekim — [sabit named T+5 kapasite sonucu](../docs/NAMED_CAPACITY_CONTROL_T5_20261010.md) doğrulandı.
Aynı 242 hedefte ham XGBoost, Ridge'e karşı +%7,444570 Naive-MAE birimi
katkı sağlıyor; blok20/60 alt sınırları pozitif. Ancak Naive kazancı
−%0,493740, ham yön %54,958678; iki aralığın %5 kazanç üst sınırları
%3,269655/%3,662587. Geçmiş shrinkage her iki kolu sıfıra indiriyor.
Bu tek, daha önce görülmüş 2023 yılına ve saat/vintage varsayımına ait
kapasite etkisi; pratik hedef veya bağımsız holdout başarısı değil.
22 piyasa+4 sentetik fit tamamlandı; rapor doğrulaması ek fit gerektirmedi.
Önce GitHub sonuç arşivinin taze replay'i kapanır; ardından tek sonraki iş,
yeni eğitim yapmadan yön–tahmin büyüklüğü hata katkısını ayırmaktır.
Aşağıdaki önceki aşamalar kronolojik kanıt olarak korunur.

11 Ekim: `named-capacity-control-t5-v1` **fit yürütmesi doğrulandı,
piyasa sonucu açık**. 22 piyasa+4 sentetik, referans Ridge yeni fit0,
48 native çıkarım; 246/242/4 korunur. Yürütme kaydı performans deneyi
tamamlandı diye kullanılamaz. [Durum ve aynı deneyin tek sonraki sıfır-fit raporu](../docs/NAMED_CAPACITY_CONTROL_T5_20261010.md).

Güncel tek iş: [sabit named T+5 kapasite kontrolü](../docs/NAMED_CAPACITY_CONTROL_T5_20261010.md).
PR39 teslimatı kapandı. 55 ilgili XGBoost/T+5 kayıt ve 117 tarif tarandı;
en yakın native kayıt CT/24 özellik kullanıyor. Aynı named label/28
özellik/2023 ortak train kontrolü için tek sabit tarif ve ayrı kimlik
hazırlanıyor. Eski Ridge native tahminleri referans; tekrar fit yok.
22 piyasa+4 sentetik tavanı; yayımlanmış ön kayıt/taze restore öncesi fit0.

Güncel sonuç: [T+5 ortak eğitim etiketi kontrolü](../docs/NAMED_LABEL_CONTROL_T5_20261010.md);44+1 fit. Doğru kontrat etiketi ham hatayı azaltıyor; Naive hâlâ geçilmiyor. Birincil label etkisi yalnız geçmiş2023 için pozitif; üretim/bağımsız holdout başarısı değil.

## Sicil

`origin-fixed-forecast-transport-v1-r2`: **sıfır-fit beta1 taşıma tamamlandı**.
Ön kayıt sonuçtan önce yayımlandı.749 origin/739 uygun fiyat/10 bilinmeyen,
5.992 türetilmiş satır. Ham T+5 Naive kazancı−%8,664216,iki aralığın üst
sınırı negatif,yıl0/3;bu sabit tarifin %5 hedefi için DECISIVE_NEGATIVE.
Aynı hedefte yeniden eğitilmiş model testi değildir; tüm kaynak/family
hakkında negatif çıkarılmaz.5.936 Decimal/32 interval/9 ret kontrolü.
R1 yüzde-birim karşılaştırma hatası korundu,R2 mutlak birimde aynı bound
ve ters normalizasyon;analiz/sayılar değişmedi. [Sonuç ve tek sonraki aday](../docs/ORIGIN_FIXED_FORECAST_TRANSPORT_20261010.md).
Ön kayıt ayrı tarihsel record'dır; eski çalışma/fit/proof'lar değişmez.

`fixed-contract-feasibility-v1`: **sıfır-fit kurulabilirlik doğrulandı**.
749 origin/14.080 aday label korunur; ikinci contemporaneous T+1/T+5
uygun739. 96 olgunluk kesimi ve 12 değişmeden aktarım kontrolü; 12'si
başarısız. Kontrat geçmişi2020'de başlar, proxy geçmişinin label'ları
aynı hedef değildir. Sekiz ret kontrolü/Decimal replay; yeni fit0.
Tahmin becerisi INCONCLUSIVE, skor yok. [Kanıt ve tek sonraki soru](../docs/FIXED_CONTRACT_FEASIBILITY_20261010.md).
Sonraki sıfır-fit tahmin taşıma ayrı ön kayıt gerektirir; eski kaynak
kabulü, gate, hedef, model veya origin değişmez.

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

### Private GitHub ve Drive taşıması

11 Ekim kullanıcı kararı: kalıcı bilimsel dosyalar private GitHub Releases'te,
Drive yalnız geçici çalışma alanı. Depo private olarak doğrulandı. İlk363
checkpoint/40.728.390 bayt taze GitHub indirmesinden hash kontrolüyle geri açıldı;
bütün Drive taşıması ve kota boşaltma henüz tamamlanmadı. Eski arşivler/deney
kayıtları değişmez. Canlı task'ın DrivePath argümanı kaldırıldı; kaynak snapshot'ı,
karar saati, altı missing ve ilk126 kilidi değişmedi.

`storage.py` standart kütüphane kullanır; model/pickle çalıştırmaz, kaynak silmez.
Paket her özgün göreli yolu, okunan gerçek bayt sayısını ve SHA-256'yı saklar;
aynı içeriği tekilleştirir. Drive placeholder boyutu kanıt sayılmaz. Credential
benzeri içerik/veritabanı yüklenmez; dışlama manifestte, özgün kopya private
yerel quarantine'de korunur ve Drive'dan silinmez. Bunlar modelin gereksiz
olduğu anlamına gelmez. Klasör yakalaması atomik piyasa/vintage kaydı değildir.

```bash
python research/storage.py pack --root output/experiments --scope EXPERIMENT --target output/evidence-staging/NEW-CAPTURE
# Yalnız aynı kaynak ve tamamlanmamış capture için açık devam:
python research/storage.py pack --root output/experiments --scope EXPERIMENT --target output/evidence-staging/NEW-CAPTURE --resume
python research/storage.py verify --archive output/evidence-staging/NEW-CAPTURE/evidence.zip --manifest output/evidence-staging/NEW-CAPTURE/manifest.json
python research/storage.py split --archive output/evidence-staging/NEW-CAPTURE/evidence.zip --manifest output/evidence-staging/NEW-CAPTURE/manifest.json --target output/evidence-staging/NEW-PARTS
```

GitHub yüklemesinden **önce** `gh repo view --json isPrivate` kontrolü yapılır.
Kaynak klasörün tamamı yüklenmez: yalnız güvenli ZIP ve manifest, helper'ın
dondurulmuş sürümü ve doğrulama makbuzu Release'e eklenir; quarantine/capture-root
yüklenmez. Aynı isimli yayımlanmış asset üzerine yazılmaz. Sonra farklı, boş bir
dizine `gh release download TAG --dir output/FRESH-DOWNLOAD` ile indirin.
İndirilen manifest/helper'ın beklenen kimliğini ve her payload'ı doğrulayın:

```bash
python research/storage.py restore --archive output/FRESH-DOWNLOAD/evidence.zip --manifest output/FRESH-DOWNLOAD/manifest.json --target output/FRESH-RESTORE
python -m unittest discover -s research/tests -v
```

Bir checkpoint'in kayıpsız taşınması beceri veya mevcut ortamda GPU inference
kanıtı değildir. Veri, scaler, source/config ve ledger bağını da koruyun. Resume
yakalanmış eski baytları kullanır; kaynak değişmediği varsayılmaz. Her silmeden
hemen önce Drive'ın **güncel** hash'i yeniden eşleşmelidir. Boş olmayan restore
hedefi ve bozuk kimlik reddedilir. Çöp kutusuna taşıma tek başına kotayı açmaz;
yalnız doğrulanan proje dosyalarının kalıcı silme aşaması ayrıca yürütülür.

Genel saklama aracı incelenmemiş SQLite dosyalarını reddetmeye devam eder.
Dört tarihsel MLflow dosyası için ayrı salt okunur inceleme yapılmıştır:
ham baytlar/hücre metinleri credential kalıplarıyla taranmış, özel Release'e
özgün dosyalar yüklenmiş ve taze indirmede SHA-256, SQLite `integrity_check`
ile bütün tablo satır sayıları eşleşmiştir. Bu sınırlı inceleme her türlü
gizli bilginin yokluğunun ispatı değildir; dosyalar herkese açık yayımlanamaz.
`reviewed-databases-manifest.json` ve `database-review-v1.py` Release'dedir;
bu manifest genel `storage.py` ZIP şeması değildir.
[Ek makbuz](evidence/private-drive-databases-delivery-20261011.json) eski
karantina kayıtlarını değiştirmez. Smoke-tracking kayıtları piyasa başarısı
değildir; current/previous kopyaları yeni bağımsız deneyler sayılmaz.

43 USDA HTML dışlaması ayrıca incelenmiştir: aynı `window.BOOMR_API_key`
alanı, [Akamai'nin sayfaya eklenen Boomerang kodundaki uygulama tanımlayıcısıdır](https://techdocs.akamai.com/mpulse-boomerang/docs/implementation).
[Yönetim API'si ayrıca kimlik doğrulama token'ı gerektirir](https://techdocs.akamai.com/mpulse/reference/api).
Yalnız bu tek alanın sınıflandırılması tarama kopyasında ayrılmış; özgün
dosyalar değişmeden `reviewed-ams-html-originals.zip` içinde korunmuştur.
Diğer credential kontrolü sürer; başka credential eklenmiş ve yanlış kaynak
kontrolleri reddedilmiştir. `reviewed-ams-html-manifest.json` ile
`ams-html-review-v1.py` ayrı private Release varlıklarıdır ve genel ZIP şeması
değildir. [Ek makbuz](evidence/private-drive-html-delivery-20261011.json)
43 taze indirmeyi doğrular; tarihsel saat/vintage veya model uygunluğu kanıtı
üretmez. Genel saklama politikasında bir anahtar izin listesi açılmamıştır.

Bir öncelikli CatBoost bağlamının üç native dosyası completed makbuzundaki
hash'lerle, veri dosyaları korunan snapshot'larla eşleşmiştir.
[İlk makbuz](evidence/private-drive-priority-native-delivery-20261011.json)
kaynak kodu bağını açık bıraktı; sonraki
[bağlama makbuzu](evidence/private-drive-priority-source-binding-20261011.json)
aynı `source_id`'yi `sources/cottonlens-research-source-v2.zip` içindeki
123 kaynak hash'iyle doğruladı. Eski makbuz değiştirilmedi. Kaynak kodu veya
model çalıştırılmadı; bu bir inference/beceri sonucu ya da bütün eski kayıp
payload'ların kurtarıldığı iddiası değildir. Tam envanterde aynı yol/hash
çifti ikinci kez sayılmaz.

[Ara completed-makbuz yedeği](evidence/private-drive-closed-receipts-delivery-20261011.json)
yalnız devam eden yakalamanın kapanmış günlük satırlarındaki, önceki
ayıklanmamış envanterde bulunmayan 11.693 kaydı korur. 12 parça taze
GitHub indirmesiyle doğrulanıp geri açılmıştır; kaynak veya capture günlüğü
değişmez. Parça manifestleri genel `storage.py` şemasıyla doğrulanabilir.
Bu bölüm native model ağırlığı, tam envanter veya silme hazırlığı değildir.
Tam yakalama ile birleşiminde aynı özgün yol/hash çifti tekrar sayılmaz.

[Birleşik kısmi geri kurma](evidence/private-drive-coherent-partial-restore-20261011.json)
26 genel arşivi, hash'i önceden incelenmiş 47 özel aslı ve tek completed
makbuzunu özgün göreli yollarında birleştirir: 17.668 dosya/2.095.998.283 bayt.
Son ağaçta her hash ve öncelikli native–makbuz–veri–kaynak bağı eşleşir.
Beş ret kontrolü geçti; eşit yol/hash tekilleşir, farklı içerik veya yalnız
harf büyüklüğü farkı reddedilir. Mevcut hedefin üzerine yazılmaz, indirilen
yedeğe hardlink kurulmaz. Model veya tarihsel kaynak kodu çalıştırılmaz.
[Tarif teslimatı](evidence/private-drive-coherent-delivery-20261011.json)
`coherent-restore-helper-v1.py`, `coherent-drive-v1-inventory.json` ve
`coherent-restore-delivery-v1.json` kimliklerini bağlar; taze private indirme
kontrol edilmiştir. Tarif önceki makbuzlardaki `fresh-*` girdilerini ve
deponun sabit `storage.py` hash'ini ister; aynı namespace'e tekrar çalışmaz.
Kısmi geri kurma makbuzu silme izni, tüm native'lerin bağ doğrulaması veya
tam Drive taşıması değildir. Büyük yakalama ve güncel kaynak kontrolleri sürer.

11 Ekim sonraki saklama kaydı: [ikinci](evidence/private-drive-closed-receipts-v2-delivery-20261011.json)
ve [üçüncü](evidence/private-drive-closed-receipts-v3-delivery-20261011.json) gruplar
14.816 ayrı özgün makbuzu daha private Release'den taze indirme/geri açmayla doğruladı.
[Geri alınabilir temizleme](evidence/private-drive-recoverable-cleanup-20261011.json)
32.476 özgün dosyanın güncel hash'e bağlı DriveFS silme işlemini kaydeder;
bulutun tam öğe sayımı bağımsız ölçülmedi. Sonraki [kalıcı temizleme/hata kaydı](evidence/private-drive-cleanup-incident-20261011.json)
insan onayı, boş çöp kutusu gözlemi, gecikmeli sekiz makbuzun ayrı temizlenmesi ve
18,72 → 14,88/15 GB kota düşüşünü kaydeder. Yalnız 0,12 GB boşluk kalmıştır;
senkronizasyon sağlığı veya yeterli alan iddiası yoktur. İki eski ilgisiz defter
aynı kimliklerle geri alınabilir çöp kutusunda tutulur; yeniden tümünü boşaltma yapılmaz.
42 başarısız Drive paketinin özgün baytları private Release'den taze geri açıldı;
bunlar kaynak/makbuz paketleridir, native eğitim ağırlığı değildir. Yakalama ve
finisher durduruldu; devam etmek için önce yereldeki kapanmış baytlar Drive'a
yazmadan ve yeni Drive indirmesi başlatmadan korunmalıdır.
Üç okunamayan/kilitli özgün korunur; önceden kaybolmuş beş küçük run ZIP'i zaten
GitHub/yerel yedektedir. Eski makbuzların `source_files_deleted=0` alanı değiştirilmez;
yeni işlem ek kayıttır. Bu, yeni fit veya tüm native payload'ların kurtarıldığı anlamına gelmez.

Sonraki [dördüncü makbuz grubu](evidence/private-drive-closed-receipts-v4-delivery-20261011.json)
2.997 ayrı özgün dosya/403.631.234 baytı yalnız mevcut yerel yakalama journal'ından
korur; yeni Drive toplu indirmesi veya fit yoktur. Önceki korumayla toplam
35.481 dosya/4.512.278.521 bayt; ayrıca korunan Kayıp Öğeler yedeği tekrar sayılmaz.
[Yeni temizleme kaydı](evidence/private-drive-v4-recoverable-cleanup-20261011.json)
2.997 güncel kaynak hash'ini ve aynı taze geri kurulmuş baytları kontrol edip
geri alınabilir silmeyi hatasız tamamladı. Sonraki [terminal kayıt](evidence/private-drive-v4-terminal-cleanup-20261011.json)
işlem anında insan onayını, boş bulut çöp kutusunu ve 14,88 → 14,51/15 GB
kota düşüşünü doğrular. Eski iki defter aynı kimliklerle önceki geri alınabilir
çöp kutusuna döndürüldü; son görünümde yalnız bu iki defter vardır. Önceki
bekleyen statüler tarihsel kayıtlardır ve değiştirilmez. Büyük model koleksiyonunun
taşınması tamamlanmadı; eski toplu yakalama/finisher otomatik yeniden başlamaz.

[GitHub evidence-20261005](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005), bilimsel dosyaları içerik hash'iyle tekilleştirir. Manifest her orijinal göreli yolu, orijinal/public SHA-256 değerini ve varlık/iç yolunu korur. Parçalar bağımsız ZIP'tir; büyük veri Git ağacına eklenmez.

Mevcut ham kaynaklar, model/checkpoint'ler, ledger'lar, tahminler, raporlar, kaynak snapshot'ları ve kesintiye uğramış çalıştırmalar korunur. Yazılım ortamı/cache, kişisel ayarlar, UI ekran görüntüleri ve içeriği güvenle yayınlanamayan veritabanları dışlanır; dışlamalar indekste görünür. Geçmişte kaybolmuş LSTM tahminleri oluşturulmaz. Yerel orijinaller silinmez/değişmez. Ayıklanmış dosya eski hash doğrulamasını geçmez ve eski deney cache'i olarak kullanılmamalıdır.

İndirme/restore komutları ana README'de. Restore araçları model/pickle çalıştırmaz. Başarısız checksum, tehlikeli yol veya mevcut hedef dosya üzerine yazma girişimi durur. Restore edilen eski bilimsel snapshot yeni kodla otomatik devam ettirilmez.

Eski `research-v2-tf-placement` için 7.331 fit ve kısmi on-call için 102 fit makbuzunun model payload kopyaları yerelde yok: toplam 22.299 dosya referansı. Makbuzlar/tahmin kanıtı korunur; bunlar yeniden kullanılabilir tam checkpoint gibi gösterilmez. Yeni zamanlama deneyinin 6.157 dosyası checksum doğrulamasıyla geri kurulmuştur. Kaybolan eski dosyalar yeni eğitimle yeniden üretilmedi.

`fas-historical-reference-witness-v1`: iki 2020 arşiv formundaki dört kod için sıfır-fit tanık denetimi; özgün hücre erişimi `BLOCKED_EXTERNAL_EVIDENCE`. Aynı PDF/form/tek boş sorgu yeni deney diye tekrar açılmaz. Kapsam ve çevrimdışı `ml/review_fas_reference.py` komutu için [tarihsel tanık raporu](../docs/FAS_HISTORICAL_REFERENCE_20261008.md).
