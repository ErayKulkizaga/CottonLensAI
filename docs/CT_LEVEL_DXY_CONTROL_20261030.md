# CT risk seviyesi ve DXY bilgisi: sabit karşılaştırma

Profil: `ct-risk-level-dxy-control-t1-v1`.

**Sonuç:** Birincil DXY karşılaştırması **INCONCLUSIVE**. Ön kayıtlı ikincil
own-level kontrolü tarihsel takip koşullarını geçti; ileri başarı henüz yok.
Aşağıdaki özgün tarif korunur; sonuç ve sınırlar en alttadır.

**Soru:** Ham DXY oynaklık durumu, Cotton'ın kendi risk seviyesi katsayısını
serbest bırakmaya göre ek tahmin bilgisi sağlıyor mu? İki kol aynı beş
parametreye sahiptir; dış sonuçlara bakarak kol, pencere veya gecikme seçilmez.

| Unsur | Kilitli karar |
|---|---|
| Veri/hedef | Mevcut CT proxy; sonraki seansın kare log-getirisi |
| Kohort | Aynı 2016–2023 / 2.006 origin; tarihler kesiştirilmez/düşürülmez |
| Referans | EWMA λ=0,94; eski 169 native dört-parametre HAR tekrar eğitilmez |
| `own_level` | Üç relative-HAR + serbest `log(origin CT EWMA)` |
| `dxy_state` | Üç relative-HAR + `log(max(raw DXY beş kare-getiri ortalaması,1e-12))` |
| Öğrenme | Aynı QLIKE/BFGS, expanding geçmiş, beş seans etiket olgunluğu |
| Seçim/refit | Geçmiş 3×63 validation; 21 gözlem refit; ağırlık 0/0,25/0,5/0,75/1 |
| Birincil | Geçmişte seçilmiş DXY kolu − EWMA QLIKE katkısı |
| Mekanizma | Ham DXY − ham own-level katkısı, aynı kapasitede |
| Bütçe | 338 piyasa + en fazla 1 native sentetik fit; tek CPU süreç/iki thread/30 dakika |

DXY istatistiği **önce ham DXY gözlem takviminde** hesaplanır. Ardından
`source_date + 1 gün 00:00 UTC <= CT_origin + 1 gün 00:15 UTC` koşuluyla
son bar alınır; en fazla üç Cotton gözlemi eskilik kabul edilir. Bu saat
**varsayımdır**: tarihsel teslim saati/ilk-vintage doğrulanmış değildir.
Geçersiz son bar atlanarak eski geçerli bar aranmaz. Kaynak tarihi, pencere
başlangıcı, availability, karar saati ve yaş tahmin payload'ında tutulur.
Eksik kaynak nedeniyle train veya OOS origin düşürülmez; kontrol durur.

Ham veri SHA-256:
`23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c`.
[Kaynak denetimi](DXY_RISK_SOURCE_AUDIT_20261029.md) bunun mevcut private
GitHub arşivinden geri kurulmuş veri olduğunu kaydeder; yeni dataset indirilmez.

## Geçmişi tekrar etmeme

Sicilde `risk-level-source-t1-v1` vardır; eski DXY deneyi saklanır. Ancak
[AMS kaynak kusuru](AMS_REQUIRED_CELL_READINESS_20261029.md) ve farklı named
hedef nedeniyle bu deney temiz CT-DXY risk testi değildir. Eski fiyat-getiri
DXY testleri de risk katkısı hükmü vermez. Son [CT şok deneyi](CT_SHOCK_FORECAST_CONTROL_20261029.md)
338 piyasa fit'iyle tamamlandı; tekrar çalıştırılmaz. Yeni çalışma eski
girdileri/sonuçları değiştirmeyen ayrı namespace ve source kimliği kullanır.

## Önceden belirlenmiş karar

Pozitif ortalama QLIKE katkısı, yıl sınırlarını koruyan paired bootstrap'ın
**hem 20 hem 60 blok** %95 alt sınırının pozitifliği ve **6/8 pozitif yıl**
gerekir. Bootstrap 10.000 tekrar, seed 42. Top%1 EWMA kaybını çıkarmak
hassasiyettir; ek başarı kapısı veya ana kohortu değiştirme aracı değildir.
QLIKE negatif olabileceği için yüzde başarı yazılmaz.

Bu koşullar geçilse bile yalnız **tarihsel takip adayı** doğar. Geçmiş yıllar
çok kez incelendi; bağımsız holdout, ileri başarı, ekonomik/işlem başarısı veya
temiz named-contract risk sonucu sayılmaz. Birincil sonuç belirsizken ham
veya ikincil kol sonradan birincil ilan edilmez. Grid/ufuk taraması/GPU yoktur.

Kaynak, hedef veya native kimlik kontrolü başarısızsa performans hükmü
verilmez. Tamamlanmış payload ve bağımsız kontrol olmadan completed yazılmaz.
Raporlamada fit girişleri kapatılır ve eski/native dosya hash'leri korunur.

## Uygulama ve teslim sınırı

Mevcut `research.ct_shock_control` ve ortak QLIKE learner/Ledger kullanılır;
ikinci eğitim motoru kurulmaz. Bağımsız kontrol DXY'yi log-difference ve
`merge_asof` ile yeniden kurar; engine'in searchsorted hizalamasını çağırmaz.
49 ilgili sentetik/regresyon testi ve Ruff geçti; bu testler piyasa başarısı değildir.

Kaynak/test/veri/ortam kilidi, private Release taze indirme/restore ve
bağımsız preflight olmadan piyasa eğitimi başlamaz. Sonuç da native payload,
eşlenmiş loss ve bağımsız replay ile private Release/draft PR'da korunur.
Saatlik Windows görevi ve üç tetikleyici kapalı kalır; ileri yayın otomasyonu
yeniden kurulmaz. Live kaynak/cohort126, eski veriler ve Drive değiştirilmez.

## Dondurulan kimlik ve ön kayıt teslimi

Source: `5846f9cbb0f360731e84f251ed311f5f665510590594645204daa95c6438ab26`.
Design: `5dd87b3d117daac2e3bac87fae9ae4b7151f339b55cb7bbc5344e347ceef7df8`.
Data: `5ad179ef26e95bcc8b8fad8252275b717c377aef09c77cc2e89dd1391aa1dfb3`.

Private [ön kayıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-level-dxy-prereg-20261030)
1.799 dosya / sıfır dışlamayla taze geri kuruldu; frozen bağımsız preflight
byte eşit geçti. ZIP SHA-256:
`d0e2f4d1df387e404659aaf38a09c6337947ae6a2f9834ddb1830ddd19a3fed9`.
Bağımsız kontrol 3.520 close, 169 geçmiş iş ve minimum 1.191 olgun train
satırı doğruladı. Etiket/saat/ilk-vintage sınırları bu teslimle değişmez.

## Sonuç — 338 piyasa + 1 native öğrenme fit'i tamamlandı

**VERIFIED:** Aynı 2.006 origin korundu; başarısız fit yok. Eski 169 model
tekrar eğitilmedi. Birincil DXY kolu koşulları geçmedi; ikincil own-level
bulgusu birincil sonucun yerine geçirilmedi.

| EWMA'ya karşı aday | QLIKE katkısı | CI20 | CI60 | Pozitif yıl | Top%1 trim |
|---|---:|---|---|---:|---:|
| Own-level ham | +0,02472399 | [+0,00619031; +0,04182102] | [+0,00694562; +0,04127647] | 8/8 | −0,00124314 |
| **Own-level geçmişte seçilmiş — ikincil** | **+0,01770104** | **[+0,00428236; +0,02989275]** | **[+0,00470139; +0,02957616]** | **7/8** | **+0,00073073** |
| DXY ham | +0,01909937 | [−0,02629838; +0,04865903] | [−0,01911088; +0,05009386] | 5/8 | −0,01190813 |
| **DXY geçmişte seçilmiş — birincil** | **+0,00177679** | **[−0,00403247; +0,00711863]** | **[−0,00333404; +0,00671255]** | **2/8** | **−0,00396780** |

Seçilmiş DXY−own-level katkısı −0,01592425; CI20
[−0,02626809; −0,00479922], CI60 [−0,02613379; −0,00469822].
DXY ham−own-level ham −0,00562462; iki CI sıfırı içerir. Bu, bütün DXY
bilgisinin yararsızlığını veya own-level'e ayrıca eklenecek DXY'nin etkisini
test etmez; aynı kapasitede bu iki temsilin yer değiştirmesini test eder.

### Somut mekanizma ve karşı kanıt

**VERIFIED:** Eski relative-HAR tarifinde relative özellikler sabitken
origin EWMA'nın tahmine etkisi zorunlu olarak birim elastisiteydi. Yeni
own-level kolunda `prediction = EWMA × exp(intercept + relativeHAR·β + γ·log(EWMA))`.
Ek γ katsayısı 169 modelin **169'unda negatif**; minimum/medyan/maksimum
−0,25387 / −0,22711 / −0,17269. Relative-HAR sabit tutulduğunda yeni
elastisite `1+γ`, yaklaşık 0,75–0,83 olur. Bu ifade absolute HAR sabitken
geçerli değildir. Model risk seviyesine daha sönümlü tepki öğreniyor.

**STRONGLY SUPPORTED:** Mevcut risk tarifinin ölçek kısıtı ve geçmiş seçimle
etkileşimi yararlı bilgiyi bastırıyordu. Yeni veri veya büyük model olmadan
serbest own-level kolu tarihsel referansı geçti. Sonuç-sonrası sıfır-fit
mekanizma denetiminde, eski dört-parametre seçilmiş kontrole karşı katkı
+0,01633534; CI20 [+0,00633268; +0,02540027], CI60
[+0,00634283; +0,02524376], 7/8 pozitif yıl. Bu ek ayrım tanımlayıcıdır;
yeni ön kayıtlı birincil test veya çoklu seçim düzeltmesi değildir.

Karşı kanıt: Ham own-level−eski ham kontrol farkının iki CI'ı sıfırı içerir.
Bu nedenle bütün iyileşmeyi tek katsayıya veya saf kalibrasyona atfetmek
doğru değildir; geçmiş shrinkage seçimleri de değişmiştir. Fiyat/yön
deneylerinin bütün başarısızlıkları bu risk bulgusuyla açıklanmış sayılmaz.

Own-level seçilmiş net kazancının **%95,915'i**, EWMA kaybı en büyük 21
origin'den gelir. En büyük tek katkı 2017-07-07 origin'i, toplamın %14,163'üdür.
Trim kazancı küçük pozitif kalır. Bu bir tek-gün sonucu değildir; ancak
seyrek büyük risk gerçekleşmelerine bağımlılık önemli bir belirsizliktir.
Ana kohort veya başarı kapısı değiştirilmedi.

### Doğrulama ve tek sonraki adım

Bağımsız kontrol 7.036 native tahmin, 10.030 Decimal loss, 16 geçmiş seçim
ve 12 ön kayıtlı CI'ı doğruladı. Tek native fixture'ın 200 tahmini bilinen
ortalama ile eşleşti; genel passed-cache kullanılmadı. Raporlama girişlerinde
fit kapalıydı; 3.871 native dosyanın önce/sonra hash'i aynı kaldı.
En büyük gradient normu 9,9046e−7. Net fit compute 49,74 saniye ve checkpoint
kopyası 6,70 saniye; bunlar toplam inceleme/teslim süresi değildir.

**Henüz doğrulanmış tahmin başarısı yok.** Görülmüş tarihsel dönemde ve
önceden tanımlı ikincil kolda bir takip adayı var. Saat/vintage ve CT proxy
sınırları sürer. Bundan sonraki tek bilimsel iş, bu own-level tarifini
değiştirmeden bağımsız ileri doğrulamanın ön kaydını hazırlamaktır; DXY grid'i,
yeni model veya otomatik görev açmak değildir. Saatlik görev kapalı kalır.

### Doğrulanmış sonuç teslimatı

Private [sonuç Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-level-dxy-result-20261030)
4.193 dosya / sıfır dışlamayla taze indirildi ve geri kuruldu. Frozen source
ile ağ/fit kapalı replay; result JSON, eşlenmiş loss CSV, bağımsız makbuz ve
mekanizma denetimini byte eşit üretti. 338 piyasa + 1 native öğrenme modeli
ve gerekli eski referanslar korunur. Eski 557 fit tarifi değişmedi; sicil
221 aşama / 560 tarif / 32.490 fit kopyasıdır. İki kolun tam-scope sicil
araması artık 169'ar completed fit bulur; bu fit'ler tekrarlanmaz.

ZIP SHA-256: `a98604a5f123cef16e566bf0e7a0432369db5838a8f259308bff4ee5d355cae8`.
Report: `2fbab7184cb8399da017b0bc5294c42c0f6fbc2706c634bc0507ce037fb571e7`.
Paired CSV: `cdc1be850dc2cea7aa49ad9b5420f1def52db2304c574a9fa82ce8421385a7df`.
Bağımsız makbuz: `d7243c16e9a5d2a469d92a7b86ca60ea994363cbfe8e2e03de752afd0d3ca570`.
Ek [sonuç kanıtı](../research/evidence/ct-level-dxy-result-20261030.json)
birincil ve ikincil bulguyu, eski kanıtların korunmasını ve taze replay'i bağlar.

Çalışma [PR #63](https://github.com/ErayKulkizaga/CottonLensAI/pull/63) içinde
incelenebilir durumdadır. Repo private kaldı; Drive, canlı model, eski cohort
ve otomatik görev değiştirilmedi. GPU eğitimi/testi çalıştırılmadı.
