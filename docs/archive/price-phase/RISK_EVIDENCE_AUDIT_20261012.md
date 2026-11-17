# Risk önceliği: mevcut tahminlerden sıfır-fit karar

## Onaylı hedef

12 Ekim kullanıcı kararı: ilk başarı hedefi güçlü EWMA referansını geçen risk
modelidir. Fiyat/yön çalışmaları ve %5 MAE, %53/%55 yön, 6/8 yıl kapıları
değişmez. Risk başarısı işlem/yön becerisi veya üretim yayını değildir.

Birincil T+1 hedefi sonraki Cotton gözleminin kare log-getirisidir; EWMA
lambda=0,94 referanstır. T+5 sonraki beş kare getirinin toplamı, yalnız
ikincil sonuçtur. T+1 incelenmiş tarihsel sonuçlardan sonra seçilmiştir;
confirmatory seçim değildir. T+30 veya ufuk/model grid'i açılmaz.

## Mevcut kanıtı yeniden eğitim olmadan denetle

`research-full-year-v1-r2` ve `interval-correction-8c61050079421507`, aynı
2016–2023/2.006 origin/ufuk kimliklerinde okunur. Eski HAR/GARCH seçimleri
değişmez. Üç geçmiş63-origin doğrulama bloğunun seçimi,21-observation refit
makbuzları, native JSON çıkarımı, eğitim frame kimliği,120-observation
warmup ve ortak T+5 etiket olgunluğu tekrar kontrol edilir. Yeni fit bütçesi0.

```powershell
$env:PYTHONPATH = 'ml/src'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python -m cottonlens_ml.research.risk_audit --experiment-root <original> --corrected-root <corrected> --output <new-namespace>
```

Mevcut ayrı CPU Python kullanılır; yerel fit izni gerekmez. Komut training,
mirror hydrate veya eski rapora yazma yapmaz. Eksik/bozuk makbuz, farklı
origin/etiket, değişen kaynak veya30-dakika bütçe ihlalinde skor yayımlamaz.

## Ölçüm ve karar

`delta = QLIKE(EWMA) - QLIKE(model)`; pozitif model lehinedir. QLIKE
`log(predicted_variance) + actual_variance / predicted_variance` negatif
olabilir; yüzde-kayıp oranı kullanılmaz. Sıfır actual geçerli; negatif actual
veya pozitif olmayan forecast geçersizdir. Yıl içi paired bootstrap10.000
tekrar/seed42/blok20 ve60. Top1% EWMA kaybı çıkarımı ve ortak-origin OHLC
vekili yalnız duyarlılıktır; ana hedef/popülasyon değiştirilmez.

- İki CI altı pozitif, ortalama pozitif ve6/8 yıl: **HISTORICAL_FOLLOWUP_CANDIDATE**.
- İki CI üstü sıfır veya altında: **FIXED_RECIPE_SUPERIORITY_NOT_SUPPORTED**.
- Diğer durumlar: **INCONCLUSIVE**; otomatik grid/reseed/yeni eğitim yok.
- Kimlik veya kausal kontrol başarısızsa performans kararı yoktur.

İlk durum yalnız ileri doğrulama adayıdır. Bağımsız ileri üstünlük olmadan
doğrulanmış başarı yok; istatistiksel katkı ekonomik/net kazanca eşit değildir.
Günlük kare getiri intraday realized variance değildir. CT=F geçişleri,
ilk vintage ve tarihsel UTC erişim bu işlemle sertifikalanmaz.

## Korunan kanıt ve ertelenen iş

[Named sentez PR49](https://github.com/ErayKulkizaga/CottonLensAI/pull/49)
private Release'den taze indirmede734 payload hash'ini doğruladı. İlk strict
byte replay yalnız JSON object sırasından başarısızdı; v2 bütün decoded
değer/record_id'leri tam, beş diğer payload'u byte-identical doğruladı.
Özgün arşiv/helper/sonuç değişmedi.52 eski tablo/39 aday etiket korunur;
eğitime kabul veya ek değerlendirme yılı sayılmaz.

Henüz başlamayan22-fit CFTC saat karşılaştırması **DEFERRED_NOT_RUN**;
tamamlanmış/negatif değildir. Risk incelemesi önce gelir. Drive işlemi,
dataset edinimi, model yayını veya PR merge bu pakette yoktur.

## Yürütme

20 risk regresyon testi ve13 mevcut full-year testi/Ruff geçti. Snapshot'ın
son origin'leri2024 hedef metadata'sı taşıyabilir; yalnız snapshot içindeki
endpoint'ler yeniden hesaplanır. Bu son origin'ler ana kohorttan zaten dışarıda;
özgün metadata değiştirilmez. Rapor, bağımsız scalar/CI makbuzu ve GitHub taze
geri kurması tamamlandığında ayrı ek kanıt kaydedilir; yeni fit0.

## Tamamlanan sonuç: üstünlük hâlâ belirsiz

482 eski inner/outer varyans fitinin eğitim kimliği/olgunluğu/native JSON
çıkarımı ve aile seçimleri doğrulandı. Yeni fit0; özgün fiyat ve varyans
tahminleri değiştirilmedi. Her ufuk2.006 origin, toplam4.012 paired-loss satırı.

| Ufuk | Ortalama QLIKE farkı | Kazanılan yıl | CI20 | CI60 |
|---|---:|---:|---|---|
|T+1|+0,01043669|6/8|[-0,00996089;0,03091330]|[-0,00673236;0,02722237]|
|T+5|+0,00341412|5/8|[-0,03057804;0,04750343]|[-0,02916342;0,04076666]|

Birincil karar **INCONCLUSIVE**: iki blokta pozitif alt sınır yok. Risk
hedefine dönmek kendi başına başarı sağlamadı; mevcut EWMA referansı korunur.
Bu, tüm risk modellerinin/kaynakların işe yaramadığı sonucu değildir.

Top1% (21origin) EWMA kaybı çıkarıldığında T+1 fark−0,00718121,
T+5+0,00894216. Bu yalnız hata yoğunlaşması duyarlılığıdır; büyük olaylar
silinmez ve ana popülasyon değişmez. T+1 küçük ortalama avantaj bu olaylara
duyarlıdır; istikrarlı üstünlük diye genellenemez.

OHLC vekili ortak kohortta T+1−0,00930830 (1.738origin), T+5−0,01456947
(1.314origin). **Aynı ortak tarihlerde** kare-getiri farkları+0,00668057/
+0,00822469: işaret farkı yalnız missing origin'lerin düşmesinden kaynaklanmıyor.
Yine de farklı vekiller aynı ekonomik hedef değildir; OHLC sonucu ilk hedefi
iptal etmez, veri bozulmasını veya intraday risk üstünlüğünü kanıtlamaz.

Bir sonraki eksik kanıt, tarif sabitken EWMA ile eşlenmiş bağımsız ileri dönem
risk tahminleridir. Bu paket yeni fit veya ileri model yayını başlatmaz.
Geçmişte aynı tarifi yeniden eğitmek belirsizliği bağımsız kanıta dönüştürmez.

## Teslimat doğrulandı

[Private Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/risk-evidence-audit-20261012)
1.300 mantıksal dosya/0 istisna içerir. Taze GitHub indirmesinde bütün hash'ler,
269 frozen ML dosyası ve report/paired-loss CSV'nin byte-identical yeniden
üretimi geçti. Bağımsız4.012 scalar kayıp/sekiz CI tekrar eşleşti; yeni fit0.
184 önceki sicil kaydı,163 kanıt hash'i,1.016 girdi hash'i ve fit sicili korunur.
33 ML +11 araştırma testi/Ruff geçti. [Sonuç](../../../research/evidence/risk-evidence-audit-20261012.json)
ve [teslimat makbuzu](../../../research/evidence/risk-evidence-audit-delivery-20261012.json),
[PR50](https://github.com/ErayKulkizaga/CottonLensAI/pull/50) içinde incelenebilir.
