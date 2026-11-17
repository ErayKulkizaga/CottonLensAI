# Neden hedefe ulaşamadık? — 6 Kasım 2026

## Karar

**Mevcut veri ve tariflerle geniş model aramasını durdur.** Yeni eğitimden önce
tam eğitim/değerlendirme girdilerinin kontrat, fiyat alanı ve zaman kapsamındaki
açığı ölç. Bu incelemede yeni piyasa/sentetik fit, veri satın alma veya indirme yok.
API/model/artifact sözleşmesi ve kapalı görevler değiştirilmedi.

Tek bir kök neden ispatlanmadı. En güçlü ortak bulgu: denenen temsillerden öğrenilen
düzeltmeler geçmişten sonraki döneme kararlı katkı taşımıyor. Ayrıca bazı kolların
kaynak kabulü ve eski karşılaştırmalarında gerçek hatalar var. Bunlar bütün
çalışmaların geçersiz veya bütün fiyat bilgisinin faydasız olduğu anlamına gelmez.
Başarılı CI ve byte eşit replay aritmetik/işletim kanıtıdır; finansal üstünlük değildir.

## Envanter ve sayım

[Makine okunur tablo](../../../research/evidence/root-cause-inventory-20261106.csv)
sicildeki **239 kaydın tamamını**, özgün sınıfları ve sonradan gelen kaynak
nitelemeleriyle korur. [Makbuz](../../../research/evidence/root-cause-inventory-20261106.json)
girdi/çıktı hash'lerini taşır; `python research/root_cause_inventory.py` üretir.
569 trial kaydı ve doğrulayıcının bildirdiği 32.934 completed-fit **kopyası**
bağımsız hipotez sayısı değildir. Tek hipotez yıllar, refit, validation, seed ve
kollarla çok sayıda fit üretir. Mevcut kayıtlar bütün insan kararlarını bağımsız
hipotez olarak saymaya yeterli değildir; yapay bir toplam verilmedi.

| Soru / aile | Gerçekte sınanan ve sonuç | Sınır / sonraki işlem |
|---|---|---|
| Legacy LSTM/TCN ve tabular/return-path | [Ortak-origin ve envanter denetimi](NAIVE_MAE_AUDIT_20261105.md): LSTM eski olumlu kıyas geri alındı; ağaç/Ridge/ElasticNet ve return-path kayıtları var | Eksik LSTM tahminlerinden paired CI üretilemez; yeniden eğitimle geçmiş tamamlanmaz. TCN learning etiketi piyasa üstünlüğü değil |
| Full-year fiyat ve risk | [Uçtan uca denetim](END_TO_END_AUDIT.md): aynı hedef ve geçmiş seçim replay; fiyat ham/seçilmiş katkı negatif | Bar tarihi gerçek yayın saati değil; risk ve fiyat ayrı |
| DXY/WTI zamanlama | [Saat deneyi](AVAILABILITY_CLOCK_RESULT_20261005.md), eski 676 native replay | Gecikme kontrolü ilk vintage kanıtı değil; aynı lag grid'i tekrarlanmaz |
| WASDE/NASS temel ve bölgesel bilgi | [WASDE](WASDE_REGIONAL_T1_RESULT_20261008.md), [NASS](NASS_REGIONAL_T1_RESULT_20261009.md) | Kaynak adı tek başına yararlı ve zamanında yeni bilgi demek değil; mevcut temsillerin sonucu bütün kaynaklara genellenmez |
| Vade eğrisi T+1/T+5 | [T+1](CONTRACT_CURVE_RESULT_20261010.md), [T+5](CONTRACT_CURVE_T5_RESULT_20261010.md): ham da hedefin altında | Aynı-vade test altkümesi temiz eğitim geçmişini kanıtlamaz |
| Loss ve fiyat kalibrasyonu | [Legacy loss](LEGACY_LOSS_LINEAGE_RESULT_20261010.md), [paired loss](PAIRED_PRICE_LOSS_RESULT_20261010.md), [kalibrasyon](PRICE_MAE_CALIBRATION_RESULT_20261010.md), [named loss](NAMED_LOSS_DIAGNOSIS_20261011.md) | MSE/MAE farkı tek açıklama değil; bütün alternatif loss'lar elenmedi |
| Price-only koşullu medyan | [Sonuç](PRICE_ONLY_MEDIAN_RESULT_20261105.md): 338 piyasa +4 kontrol; seçilmiş −%0,082523, 0/8 yıl; ham −%0,118619 | Bu sabit tarif için geçerli STOP; bütün fiyat modellerinde sinyal yokluğu değil |
| Küçültme ve zayıf sinyal | [Ridge kontrol](WEAK_SIGNAL_RESULT_20261009.md), [kararlılık](SELECTION_STABILITY_RESULT_20261010.md) | Seçim küçük etkileri azaltabiliyor; gerçek ham piyasa sonuçları da kötü. Ağırlığı kaldırmak desteklenmiyor |
| XGB zayıf sinyal | [132 sentetik fit](NAMED_WEAK_SIGNAL_EXECUTION_20261012.md): üç seed ham beklenen %8,38/%8,64/%8,50 | En az bu öğretici öğreniliyor; bir seed gerçekleşmiş etki %1,02. Gerçek piyasa gücü veya evrensel yeterlik değil |
| Named risk/HAR | [Kaynak kusuru](RISK_LEVEL_SOURCE_20261029.md): üç profile yayılan lifecycle qualification | Temiz aynı-kontrat becerisi veya sinyal yokluğu çıkarımı askıda |
| Risk zayıf öğretici ve ileri test | [1.044 sentetik fit](RISK_WEAK_TEACHER_20261029.md): ortalama oracle kurtarma %76,40; [güncel durum](STATUS.md): yakın dönem üstünlüğü belirsiz, toplama kapalı | Sentetik risk öğrenmesi fiyat başarısı değil; bağımsız tamamlanmış ileri üstünlük yok |
| CFTC/FAS/hava/diğer kaynaklar | CSV'de exploration, admission, diagnostic ve fit kayıtları ayrı | İnceleme veya reddedilmiş kaynak tamamlanmış tahmin deneyi sayılamaz; bu aileler topluca elenmiş değil |

## Nerede yanlış yaptık?

1. **Eşit olmayan karşılaştırma:** eski LSTM'de farklı origin kapsamlı Naive ile
   kıyas olumlu izlenim vermiş. Ortak-origin düzeltmesi bunu geri alıyor.
2. **Kaynağı erken kabul:** resmi AMS yıllığında parser ham kimliği doğru okusa
   bile kontratın yaşam döngüsü denetlenmemiş. 218 history/217 eval tarihte geçmiş
   teslim ayı var. Sonraki günlük tanıklar gerçek feature kusurlarını da gösteriyor;
   yalnız metadata kozmetiği değil. [Gerekli hücreler](AMS_REQUIRED_CELL_READINESS_20261029.md).
3. **Kontrolleri fazla genelleme:** native replay, test geçişi veya güçlü sentetik
   kontrol zayıf piyasa sinyalinin öğrenilebildiğini ispatlamaz. Kaynak hatası varken
   aritmetiğin doğru olması bilimsel çıkarımı kurtarmaz.
4. **Tekrarlanan tarihi yeni kanıt sanma riski:** 2016–2023 ve incelenmiş 2024+
   tekrar kullanıldı. Model başına CI, bütün insan/model aramasını düzeltmez.
5. **Maliyet ile bilgi kazanımını ayırmama:** fit/PR/belge sayısını ilerleme ölçüsü
   yapmak, aynı açıklamaya küçük varyasyonlarla dönmeyi kolaylaştırır. Kayıtlar
   toplam token/fatura dökümü içermiyor; maliyet veya israf yüzdesi uydurulamaz.

İlk dört başlık belgeli teknik/metodolojik sorun veya çıkarım sınırıdır; beşinci
başlık araştırma yönetimi değerlendirmesidir. Güncel pipeline bu hataların hepsini
hâlâ yapıyor iddiası değildir; uçtan uca denetimde pek çok sınır zaten düzeltildi.

## Hatanın yayılımı ve API ayrımı

Lifecycle düzeltmesinin açıkça etkilediği profiller:
`ams-named-ewma-har-full-year-t1-v1`, `ams-risk-curvature-score-t1-v1`,
`risk-level-source-t1-v1`. CSV bu kayıtları özgün kararlarını silmeden niteliyor.
Günlük/haftalık onarım kısmi; tam eğitim girdisi kabulü ve performans etkisi
ölçülmüş değil. Diğer named kolların tamamı otomatik temiz veya bozuk ilan edilmedi.
Bu hatanın CT=F fiyat-only çalışmasına geçtiğine dair kanıt yok.

Zincir: sağlayıcı → cache/raw → parser → tarih/kontrat eşleme → train-only
özellik/hedef → native model → fiyat dönüşümü → rapor. Son medyan çalışmasında
`ml/src/cottonlens_ml/research/naive_mae.py::_inputs` yerel Parquet okur; backend
HTTP yanıtı fit'in girdisi değildir. Bu nedenle uygulama API kesintisi bu frozen
sonucu açıklamıyor. Sağlayıcı doğruluğu ise raw dosya hash'i ile ispatlanamaz.

Denetimde boş/yanlış sembollü Yahoo yanıtının cache'i bozabilmesi düzeltildi;
bu arızanın eski frozen dataset'te gerçekleştiği gösterilmedi. Ortak hedefte
eksik T+1 ve kronolojik olmayan pencere sınırları da düzeltildi; ana frozen
sonucun bunlardan etkilendiği gösterilmedi. HTTP429 kaynak kapsamını engelleyebilir,
ama tek başına modelin negatif MAE'sinin nedeni değildir.

CT=F'deki high–low dışı kapanışlar doğrudan bozuk etiket sayılmaz: 997 tarihli
bağımsız tablonun 996'sı eşleşmiş. Roll bazı etiketleri değiştiriyor; aynı-kontrat
test yollarında da başarısızlık var. **Eğitimdeki roll kontaminasyonunun dolaylı
etkisi dışlanmış değil.** [Fiyat semantiği ve karşı kanıt](PRICE_SEMANTICS_AUDIT_20261005.md).

## Veri mi az, model mi yetersiz?

Mevcut kanıt bunların sayısal payını ayıramıyor. T+1 signed ACF1 yaklaşık0,0745,
absolute ACF1 yaklaşık0,2607: hareket büyüklüğünün kümelenmesi yön sinyali değildir.
T+5 örtüşmesi bağımsız örnek sayısını azaltır. 3×63 seçim bloklarının kazanan farkı
küçük ve kararsız; 63 gözlemde blok60'ın yalnız dört başlangıcı vardır.
Bu koşullarda daha çok aynı veri, daha çok feature ve daha büyük modelin etkileri
ayrı deneyler olmadan nedensel olarak ayrıştırılamaz.

En son Quantile'da hem ham hem seçilmiş sonuç negatif; iki seçilmiş CI üst sınırı
%5'ten çok uzak. Bu tarifi büyütmek için gerekçe yok. Eski XGB ve risk kontrolleri
motorun bazı zayıf öğreticileri öğrendiğini gösteriyor; bütün nonlinear etkilerin
veya yalnız %5 büyüklükte piyasa etkilerinin yeterli güçle sınandığını göstermiyor.

## Neyi henüz güvenilir biçimde sınamadık?

| Açık soru | Mevcut cevap | Tek sonraki ayırıcı adım / kabul şartı |
|---|---|---|
| Tam train+test boyunca kimliği ve erişim zamanı kanıtlı hedefte fiyat becerisi? | Tam kapsam kanıtı yok. Temiz test altkümesi, sonradan günlük tanık veya 2008–2009 risk deneyi bunun yerine geçmez | Yalnız mevcut dosyalardan gerekli tüm price/feature bağımlılıklarının coverage matrisi; her hücreye kontrat, field, tarih ve availability tanığı. Eksikse fit yok |
| Mevcut örneklem gerçekçi %5 sinyali ayırt edebilir mi? | Bazı öğreticiler öğreniliyor; evrensel piyasa güç hesabı yok | Önce mevcut güç kontrollerinin hedef/teacher/noise/cohort uyum tablosu. Yeni kontrol ancak belirli açık ve ön kayıtla; otomatik seed veya fit yok |
| Zamanında erişilebilir ek bilginin farklı temsili faydalı mı? | Bazı temsiller denendi; kaynakların tamamı elenmedi | Bir ekonomik mekanizma, eski tarife somut fark ve first-vintage kanıtı gösterilmeden kaynak/model seçme |
| Bağımsız ileri başarı var mı? | Tamamlanmış kanıt yok | Tarihsel analizden ayrı yeni ön kayıt gerekir; eski dönem yeniden holdout yapılamaz. Kapalı görev kendiliğinden açılmaz |

## Harcama kararı ve tamamlanma kontrolü

**Öneri: önce belirli veri açığını ölç; mevcut tarif aramasını durdur.** Bir
sonraki olası çalışma eğitim değil, mevcut arşivden tam bağımlılık coverage'ıdır.
Kabul: tam train/test kapsamı ve tüm çözümsüz kimlik/zaman çatışmaları listelenmeli.
Eksik kaynak yokmuş gibi doldurulamaz. Bu çalışma sonucunda tam kapsam mümkün
değilse satın alma/indirme kararı için somut eksik listesi çıkar; maliyet ve kalite
kanıtı olmadan veri yatırımı önerilmez. Başarı hedefi değişmedi.

Bu rapor ve bağlı CSV, seçilen sıfır-eğitim teşhis planının teslimatıdır. Sicil
doğrulaması239/569/32934 verdi. Envanter239 benzersiz kayıt kimliğini kapsıyor;
girdi hash'leri eski sicillerin değişmediğini doğrulamak için tutuldu. Kontrol
mekanizmalarının eski sonuçları kullanıldı; yeni bootstrap/native replay veya
bütün CI tekrarlanmadı. Yeni kod yalnız standart kütüphaneyle tablo üretir.
Eski arşivlerin tamamı bu turda tekrar indirilmedi; bulguların kaynağı bağlı
denetimler ve sicildir. Nedensel paylar, toplam token maliyeti ve tam veri
zaman kapsamı bilinmeyen olarak bırakıldı. Bunları bilmemek teşhisin sınırıdır;
yeni bir modelin bunları çözeceği varsayılmadı.
