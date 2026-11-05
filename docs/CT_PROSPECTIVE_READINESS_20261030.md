# CT risk adayının ileri doğrulama kurulabilirliği

Profil: `ct-own-level-prospective-readiness-v1`. **Sıfır yeni fit, sıfır yeni
dataset, sıfır ileri tahmin.** Bu rapor bir ileri doğrulama ön kaydı değildir.
[PR63'teki aday](CT_LEVEL_DXY_CONTROL_20261030.md) korunur: birincil DXY
belirsiz, ön kayıtlı ikincil own-level tarihsel takip koşullarını geçmiştir.
Bağımsız ileri başarı henüz yoktur.

## Ne değişti?

**VERIFIED:** Mevcut `research.prospective` akışı 126 Naive/EWMA origin'i
veya eski fiyat modelleri içindir; bu öğrenilmiş risk modelinin doğrulaması
değildir. `live.capture_market` yalnız iki yıllık snapshot toplar. Bu snapshot
2010'dan başlayan expanding geçmişin yerine geçirilemez veya EWMA baştan
başlatılamaz. Mevcut saatlik görev kapalı kalır; hiçbir toplama çalıştırılmadı.

**VERIFIED — tam geçmiş:** Eldeki checksum bağlı ham dosyada 2010-01-04–
2026-09-22 arasında 4.205 Cotton barı vardır. Eski native geçmişin ilk 3.520
fiyatı birebir eşleşir; risk özellikleri ortak builder ile yeniden kuruldu.
Eski train warmup'ı index119, 2010-06-24'tür; yalnız 22 hazır HAR gözlemi
var diye train başlangıcı değiştirilemez. 2024+ burada yalnız gelecekteki
eğitime başlangıç verisidir; bağımsız holdout veya yeni performans skoru değildir.

Yeni `append_snapshot`, seed'in son seansıyla örtüşme ister; bilinen örtüşmede
fiyat revizyonunu, tarih ekleme/silmesini ve kesintisiz geçmişi iki yıllık
pencereyle değiştirmeyi reddeder. İki özgün kaynak korunur. Bu kontrol
**yalnız bilinen örtüşmeyi** doğrular; yeni gelecekteki seansların eksiksizliği
ve kaynak makbuzunun karar saatinden önce oluşması ayrıca doğrulanmalıdır.

## 126 origin neden başarı vaadi değildir?

Yeni adayın eski 2.006 eşlenmiş loss farkı kullanıldı; yeni model seçilmedi.
Pooled ortalama +0,01770104. Yıl sınırlarını aşmayan Bartlett kovaryans ölçeği
blok20 için 0,08882734; blok60 için 0,10373231.

**Yalnız koşullu normal yaklaşım:** bu etki ve kovaryans gelecekte aynen
sürerse, tek bir %95 aralığın alt sınırının pozitif çıkması için:

| Varsayım | Blok20 | Blok60 |
|---|---:|---:|
|126 origin'de yaklaşık saptama olasılığı|%9,80|%8,96|
|%80 tek-aralık olasılığı için yaklaşık origin|2.226|2.599|

Bunlar **kalibre edilmiş güç, iki CI'ın ortak başarı olasılığı, zorunlu süre
veya bekleme kararı değildir**. Tarihsel seçimi, nadir olay dağılımını ve
gelecek rejim değişimini düzeltmez. Tarihsel alt-CI büyüklüğünü yalnız ayrı
bir senaryo sayınca gerekli sayı daha da artar; o alt sınır geleceğin
etkisine güvence değildir. Gerçek etki sıfırsa daha uzun beklemek üstünlük yaratmaz.
Eski [ölçüm kurulabilirliği](RISK_MEASUREMENT_READINESS_20261012.md) farklı HAR
tahminleri için aynı tür güç sınırını göstermişti; burada yeni own-level
adayının farklı loss kovaryansı hesaplandı, eski fit/label transport'u tekrarlanmadı.

Tanımlayıcı 126-origin pencerelerin %71,77'sinde ortalama katkı pozitiftir.
Bu 1.881 **örtüşen** pencere bağımsız deney veya güç hesabı değildir. Pozitif
ortalama ile pozitif güven alt sınırı birbirine karıştırılmaz. 126-origin
eski baseline kilidi değiştirilmedi; başarı eşiği gevşetilmedi.

## Kontrol, başarısız ilk denetim ve kapsam

14 test/Ruff geçti. Bağımsız doğrulama 2.006 Decimal paired loss farkını,
20 normal senaryoyu, 6.139 tanımlayıcı pencereyi ve 4.205 seed satırını denetledi.
Kovaryans bağımsız olarak yıl içinde sıfır-dolgulu blok toplamlarının
karelerinden kuruldu; engine'in lag çarpımları çağrılmadı.

İlk denetim (`execution-v1`, source `1dcc528b…`) başarısızdır. Doğrulayıcıda
iki eşit tahminin QLIKE farkını ayrı büyük terimlerden çıkarmak Decimal
yuvarlamasıyla çok küçük sahte işaret üretiyordu. `ln(b/m)+y(1/b−1/m)`
eşlenmiş kimliği ve tam eşitlik regresyon testiyle düzeltildi. Yeni namespace
`execution-v2`, source `b80e0971…`. İlk snapshot/sonuç korunur ve tamamlanmış
kanıt sayılmaz. V1/V2 ana planlama değerleri ve özgün girdi hash'leri aynıdır;
piyasa tahminleri, hedefler ve eski deneyler değişmedi.

Bu aşama **INCONCLUSIVE — yalnız ileri kurulabilirlik**, piyasa performansı
hükmü değildir. İleri kohort henüz kilitlenmedi. Sonraki tek uygulama işi,
bu tam geçmişi ve değişmeyen own-level tarifini kullanan **manuel** risk
kayıt protokolüdür: örneklem bütçesi ve düşük güç sınırı önceden açık olacak,
kaçan origin yerine geçmiş tahmin konmayacak, sonuç görülerek süre uzatılmayacak.
Saatlik görev, otomatik yayın, yeni model veya grid açılmaz.

## GitHub teslimatı

Private [kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-risk-prospective-readiness-20261030)
601 dosya / sıfır dışlamayla taze indirildi ve geri kuruldu. İndirilen
frozen source ile result, ready, loss CSV, seed Parquet ve bağımsız makbuz
fit/ağ kapalıyken byte eşit yeniden üretildi. Başarısız ilk revision da
aynı pakette korunur. ZIP SHA-256:
`6bf1743c532dc1d71d8c30f7949c8e1153526d44c726ed2ccaf1339311bf3806`.

Analiz kaynağı `b80e097158cd06104648ecac96f21ee0278313935b9904ccdc496d05ea238fdd`;
bu yeni bir eğitilmiş model kaynağı değildir. Adayın fitted source'u
`5846f9cb…` kalır. Ek [sicil kanıtı](../research/evidence/ct-risk-prospective-readiness-20261030.json)
önceki 221 aşamayı, 560 fit tarifini ve fit sicilinin byte hash'ini korur.
Yeni sicil 222 aşamadır; tamamlanmış eğitim sayısı değişmedi.
