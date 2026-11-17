# CT sekiz yıllık EWMA-relative HAR sonucu

**INCONCLUSIVE — EWMA üstünlüğü gösterilmedi.** Ön kayıtlı tek dört parametreli
QLIKE-HAR adayı169piyasa+1sentetik fit ile kapandı. Bütün2.006origin korunur;
yerel CPU/tek proses/iki thread. Parametre, kaynak, λ veya işaret grid'i yoktu.

| Karşılaştırma (baseline−candidate QLIKE) | Kazanç | CI20 | CI60 | Pozitif yıl | Top1% trim |
|---|---:|---|---|---:|---:|
| Ham sabit aday | +0,01747463 | [−0,02077932;+0,04345457] | [−0,01466282;+0,04386211] | 4/8 | −0,01183509 |
| Geçmiş validation seçimi | +0,00136570 | [−0,00458287;+0,00699814] | [−0,00453374;+0,00670574] | 4/8 | −0,00499849 |

İki CI sıfırı içerir;6/8yıl ve pozitif trim koşulları da sağlanmadı.
Ham pozitif ortalama başarı değildir.2022ham katkısı+0,10457348 olup,
validation o yıl weight0seçti. Bu geleceği bilen bir seçimle değiştirilemez.
Yeni model kapasitesi/λ/recency grid'i veya Colab başarısı ilan edilmez.
Fiyat5%MAE/53–55direction kapıları bu risk ölçümüyle geçilmiş sayılmaz.

## Değişmeyen kapsam ve yürütme

Orijinal risk-audit CT sürekli fiyat kare-getiri/EWMA0,94 ve2016–2023
kapsamıdır; named settlement veya fiziksel risk ölçümü değildir. Mevcut
origin/target endpoint'leri aynıdır. Model yalnız1/5/22gün HAR bilgisini
EWMA'ya relatif öğrenir; offset bilinen origin state'tir. Modelde4parametre,
mevcut convex QLIKE/BFGS ve no-fallback sertifikası vardır.

Son169işte mintrain1.191, recent189validation/3×63, warmup119,
`target_date_5<cut` ve21Cotton-observation refit korundu. Seçim yalnız
geçmiş iç bloklarda0/.25/.5/.75/1aritmetik variance mixture ile yapıldı;
küçük weight tie kuralı değişmedi. Primary selected risk kapısı ön kayıtlıydı.
Asimetri/CFTC başarısız screen'leri girdilere eklenmedi.

V1/V2 hazırlıkları hiç fit başlamadan superseded olarak korunur. Ledger
callback'i explicit`before_compute=` ile bağlandı; oturum dolduğunda market
compute'a girilmediğinin regresyon testi geçti. Native identity bütün planı
`design_id` ile bağlar;169makbuza386bin tarih tekrar kopyalanmaz. Altı yeni
test ve ilgili20model testleri geçti; Ruff/diff check temiz. Gerçek model
compute toplam9,52s ve checkpoint1,72s ölçüldü; makine/koşu özeldir.

169native modelin optimizer success/standardized gradient sertifikası geçti;
en yüksek∞norm9,9482e−7. Tahminler native coefficient'tan tekrar kuruldu.
Source id:`6ccd84e5e81a451b90702bf6c69e99c8154bab5420aac56ec859c08c3f07e777`;
design:`000d435ba76a99943d52ee8a5fd4a2d121f39b802338f313f32111c6505d508a`;
data:`013f58e5804b29435f32161ba1cf42ac2fc56046f646f4993294bf267079e4f7`.
Ön kayıt298üye taze private GitHub restore ile eğitimden önce doğrulandı.

## Bağımsız denetim ve sınır

Raw close'tan Decimal ile10.532HAR scalar ve2.006actual/EWMA origin,
6.018QLIKE loss kuruldu.169cutoff/386.182train satırı,3.518native forecast,
sekiz geçmiş weight seçimi ve dört loss CI ayrı sum-difference bootstrap
ile doğrulandı. Sentetik native mean ayrıca replay edilir; piyasa kanıtı değildir.
Yeni report/verify fit0.

Historical source clock/final-vintage varsayımı ve seen research history
sınırı korunur;2024+selection yoktur, independent holdout yoktur. Çoklu
araştırma kararlarına göre düzeltilmiş başarı iddiası yoktur. Eski169fitin
yeniden eğitimi veya mevcut sonucu değiştirme yerine native release'i geri kur.

Yerel çalışma:`output/ct-ewma-har-control-20261013/execution-v3/`.
Ön kayıt:`ct-ewma-har-preregistration-20261013` private Release.
Sonuç:[private Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-ewma-har-result-20261014),
1.499 dosya/SHA256 `34df6cd4f114364164e3c1dac4ac6f218c56d25d79355bb42cd5dc7cb29b2aef`.
Taze indirme/restore doğrulandı. Ayrı Python 3.13 ortamındaki zero-fit bağımsız
replay makbuzu paket kopyasıyla byte-identical;169 market+1 synthetic native
fit,3.518 tahmin,386.182 train satırı,4 CI yeniden doğrulandı.
[Sonuç kanıtı](../../../research/evidence/ct-ewma-har-result-20261014.json)
sicile INCONCLUSIVE olarak işlendi:202 aşama/543 tarif/30.244 fit kopyası.
Tam öneri kontrolü `REPEAT_FIT_RECIPE`; aynı fit tekrar edilmez.
Mevcut EWMA korunur; hedef measured superiority bulunana kadar aktiftir.
