# Named T+5 zayıf-sinyal kontrolü — mevcut tasarımın yürütülmesi

PR44 yalnız girdileri doğruladı, öğrenici fit0. Bu teslimat aynı tasarımı
`Experiment.fit`/Ledger yoluna bağlar; yeni model, grid veya piyasa fit'i değildir.
Eski readiness, kaynak/kapasite/loss deneyleri değiştirilmez ve tekrarlanmaz.

## Kilitli soru ve sınırlar

Mevcut sabit XGB, origin'de bilinen tek shallow-split öğreticinin pratik
büyüklükteki etkisini mevcut eksiklik, gürültü ve örnek miktarında çıkarabiliyor mu?
Üç seed703/1703/2703 × null/injected;32 alan, geçmiş3×63 iç seçim,
246 dış origin/242 uygun/4 bilinmeyen,100 ağaç/depth2/eta.03/minchild20.
Toplam132 sentetik fit, piyasa fit0; bir süreç/iki thread,30 dakika oturum.
Fiyatlar tarihsel referans, etiketler semi-synthetic; tutarlı traded futures
patikası veya bağımsız piyasa holdout'u iddiası yoktur.

Öğretici/threshold/beta/sigma ilk528 olgun satırdan sabitlenmişti; değiştirilmez.
Oracle yaklaşık%10 beklenen kazanç, öğrenci başarısı değildir. Sentetik
gürültü beş gelecek günlük innovation toplamıdır; yalnız olgunluk kurallarına
uyan geçmiş etiketler eğitime girebilir. Feature whitelist'e oracle/noise/
gelecek etiket eklenmez. Preprocessing ve hedef ölçeği her fit'in geçmişinden gelir.

## Ön kayıt ve doğrulama

Kaynak/veri/ortam/design/rapor kimliği dondurulur; yeni namespace gerekir.
Mevcut readiness senaryoları bayt düzeyinde yeniden üretilir; native öğrenici
ve feature/target kodu eski snapshot ile eşleşir. Taze GitHub ön kayıt geri
kurması olmadan `prepare` eğitim deposu oluşturamaz. Drive mirror kullanılmaz.

Yürütme `python ml/full_year_cpu.py` üzerinden aynı engine'e gider:
`prepare → pilot-plan → pilot → report`. Profil
`named-weak-signal-control-t5-v1`; açık registration/decision-contract ve
ayrı CPU ortamı seçilir. Bütçede duruş checkpoint'i korur; başka kimliğin
cache'i veya yarım makbuz completed sayılamaz. Ek öğrenme kontrolü fit'i yoktur;
bu altı senaryo zaten kontrolün kendisidir.

Ham ve geçmişte seçilmiş tahminler, origin/target/eligibility, ölçek,
öğretici/noise ve seçilmiş ağırlık korunur.132 native modelin tahminleri,
yalnız geçmişten yeniden hesaplanan preprocessing/scaler ile tekrar çıkarılır.
Bağımsız doğrulayıcı Gaussian quadrature ile beklenen MAE'yi ve scalar geçmiş
shrinkage seçimini denetler; yeni fit yapmaz. Ham/selected yön ve aktiflik,
gerçekleşmiş MAE ve log/standardized MSE ikincil betimsel sonuçlardır.

## Önceden belirlenen karar

Birincil ölçüt **ham öğrenicinin bilinen koşullu dağılımdaki beklenen dış
fiyat-MAE kazancı**; geçmiş shrinkage ve gerçekleşmiş noise kaybı ayrı gösterilir.

- Üç injected seed ≥%5 ve her null kazancı ≤0: yalnız bu basit öğretici için
  öğrenme yeterliği desteklenir; mevcut bilgi/temsil daha güçlü şüphelidir.
- Üç injected seed <%5 ve null kontrolü sağlamsa: bu sabit düzen yararlı
  bilinen etkiyi yeterince çıkaramıyor; kaynakta sinyal yokluğu çıkarılamaz.
- Karışık seed veya başarısız null: belirsiz; otomatik reseed/grid yok.

Üç seed kalibre edilmiş istatistiksel güç tahmini değildir. %5, T+1%53/T+5%55,
6/8 yıl ürün kapıları değişmez; sentetik başarı piyasa başarısı yerine geçmez.
Sonuçtan önce yeni piyasa verisi/model/loss/grid veya canlı126 skor açılmaz.

## Ön kayıt R2 — fit öncesi sınır düzeltmesi

R1 kaynak/ön kayıt değiştirilmeden korunur; fit0. Bir bilinmeyen origin'de
referans fiyatı da NaN: NaN–NaN karşılaştırması aynı eksikliği kabul eder,
fiyat/gerçekleşen hedef doldurulmaz. CSV okuyucu `null` kol adını eksik değer
saymamalıdır; yalnız boş alanlar NA kabul edilir. R2 bu rapor sınırlarını
regresyon testiyle korur. Öğretici/model/cohort/132-fit bütçesi değişmez.

## Tamamlanan sonuç —132 sentetik fit, piyasa fit0

**VERIFIED:** bilinen basit zayıf sinyal ham öğrenicide üç seed'de de %5
beklenen fiyat-MAE sınırının üzerinde çıkarıldı.132 native replay,1.476
orijinal origin satırı/1.452 uygun/24 bilinmeyen, bağımsız quadrature ve dört
coherent rapor bozulması reddi doğrulandı. Eksik origin/target/fiyat doldurulmadı.

| Noise seed | Ham beklenen kazanç | Seçilmiş beklenen kazanç | Ham gerçekleşmiş kazanç | Seçilmiş gerçekleşmiş kazanç | Geçmiş ağırlık |
|---|---:|---:|---:|---:|---:|
|703|%8,376338|%8,0804|%8,538581|%8,150313|0,75|
|1703|%8,644082|%8,6329|%6,497799|%7,992894|0,75|
|2703|%8,500825|%3,7696|%1,021117|%1,717755|0,25|

Null ham beklenen kazançlar −%2,703662/−%1,719053/−%2,083850;
seçilmiş ağırlık üçünde0, beklenen/seçilmiş kazanç0. Önceden kilitli karar
`SIMPLE_WEAK_TEACHER_RECOVERY_SUPPORTED`. Kanıt sınıfı **DECISIVE_POSITIVE**
yalnız bu sabit öğreticiyi öğrenebilme içindir; piyasa performansı değildir.

**STRONGLY SUPPORTED:** aynı düzen en az bu basit yararlı bilgiyi çıkarabilir;
“öğrenici herhangi bir zayıf sinyali öğrenemiyor” açıklaması bu kontrolü
geçmiyor. Bu, bütün nonlinear/koşullu piyasa etkileri için yeterlik kanıtı değildir.

**VERIFIED:** aynı seed2703'te beklenen ham %8,50 etki gerçekleşmiş 242
etikette yalnız %1,02 görünür; geçmiş shrinkage beklenen etkiyi %3,77'ye indirir.
Tek yıllık gerçekleşmiş negatif/küçük sonuç, kendi başına kaynakta sinyal
olmadığını kanıtlamaz. Üç seed bağımsız piyasa yılları veya güç kalibrasyonu değildir.
Öte yandan eski gerçek piyasa **ham** numeric/mask sonuçları da Naive'ın
gerisindedir: yalnız shrinkage'ın kaldırılması piyasa başarısını açıklamaz.

132 fit'in compute süreleri toplam46,30 saniye, checkpoint kopyaları3,40
saniyedir; toplam oturum/replay/kimlik doğrulama süresiyle karıştırılmaz.
Bu çalışmada GPU darboğazını destekleyen kanıt yoktur.

## Sonraki tek iş — sıfır-fit bilgi kapsamı denetimi

Önce sonuç arşivinin taze GitHub native/scalar replay teslimatı kapatılır.
Ardından mevcut named tarifine gerçekten hangi bilgi kaynaklarının girdiği,
origin'deki güncellikleri/eksiklikleri ve önceki ablation'ların **aynı hedef ve
bilgi kümesini** test edip etmediği salt okunur eşlenir. Mevcut32 alan:
14 Cotton proxy fiyat/volume/range,8 çapraz-piyasa getiri/korelasyon,2 takvim,
4 curve/erişilebilirlik ve4 aynı-kontrat geçmiş getiri. NASS/FAS/CFTC/weather
bu tarifin feature listesinde yoktur; bu kontrolden o kaynakların işe yaramadığı
sonucu çıkarılmaz. Yeni veri/model/grid veya piyasa fit'i otomatik açılmaz.

R1 ve R2 ön kayıtlar korunur. R2 ön kayıt arşivinin yardımcı Markdown metnindeki
karakter kodlaması daha sonra güncel belgede düzeltildi; ML/veri/decision-contract
kimlikleri değişmedi. Sonuç arşivi düzeltilmiş belgeyi ayrıca taşır. Eski148
kanıt dosyası ve526 fit-tarifi kaydı değiştirilmedi; yeni132 fit sicilde yalnız
sentetik olarak sınıflandırıldı. Genel legacy role indeksinin null/injected
adlarını market sayması bu yeni indeks projeksiyonunda düzeltildi; özgün
makbuzlar ve eski sicil değişmedi.

## Taze GitHub sonuç teslimatı

[Sonuç Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/named-weak-execution-result-20261012)
14.012.064 bayt; SHA256`3c72d56735b00e1b40c027eb4b1310e2b00e605f1db770004d4886fc15f2b9fc`.
1.760 payload üyesi taze indirmede eşleşti; frozen source132 native replay ve
1.452 scalar/quadrature kontrolünü ek fit olmadan yeniden üretti. Tahmin,
ready ve rapor baytları özgünlerle aynı.179 önceki aşama/526 tarif/148 eski
kanıt değişmedi. PR46; merge/model yayını/Drive silme yapılmadı.
