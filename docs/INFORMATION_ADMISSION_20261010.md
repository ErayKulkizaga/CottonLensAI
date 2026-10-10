# Mevcut bilginin sıfır-fit kabul elemesi — 10 Ekim

[Karar eki](RESEARCH_PRIORITY_ADDENDUM_20261010.md) P4 kapsamı;
[kayıtlı loss sonucu](PAIRED_PRICE_LOSS_RESULT_20261010.md) ve GitHub restore
doğrulaması tamamlandıktan sonra. Yeni model/grid/veri edinimi, fiyat düzeltmesi
veya kaynak kabulü yapılmadı. [Sayısal kanıt](../research/evidence/information-admission-20261010.json).
Başlangıç commit'i `e64b72c9d281c233d62f517e6073e3fa510e2c39`; ML kaynak
`d3ca00e4…` değişmedi. Eski sicil ve plan silinmez.

## Gözden kaçmış somut ayrım

**VERIFIED:** mevcut AMS `FUTURES TODAY` tabloları yalnız Close doğrulama
verisi değil; gerçek vade adlarıyla aynı raporda **10 farklı kontrat fiyatı**
içeriyor. 998 ham belgenin 997'sinde okunabilir referans tarihi, toplam
**9.970 fiyat** var. 2020-01-02–2023-12-29; bağımsız token/Decimal okuyucusu
ile mevcut `price_reconciliation.parse_quotes` 997 tabloda birebir eşleşti.
1.000 eski girdi (998 belge, review, market) önce/sonra checksum doğrulandı.

Eski `ams_exploration.add_quotes` özelliği
`spot_41_4_34 / cotton_close - 1`: **spot–futures farkıdır**. Bu, yakın ve
sonraki vadeli kontratların farkı değildir. Eski `quote_L1/L2/L6` T+5 deneyi
246 adet 2023 origin'inde seçilmiş tahminleri sıfırlamıştı; L1 ham Naive
kazancı −%1,434923. Sicilde INCONCLUSIVE; bundan T+1 veya gerçek vadeler
arası farkın işe yaramadığı çıkarılamaz. Eski sonuç tekrarlanmadı.

Güncel 24 özellik gerçek vade fiyatlarını içermez. Sicil/fit özelliklerinde
`contract_curve`, `term_spread`, `curve_slope` eşleşmesi yok; ilgili kaynak
modülleri incelendi. **Bu, indekslenmemiş eşdeğer deneyi dışlayan özgünlük
ispatı değildir.** Tam kaynak/clock/cohort kararı olmadan eksik `--proposal`
bir tam tarifmiş gibi üretilmedi; yeni fit kapısı açılmadı.

## Kabulü engelleyen gerçek sınırlar

| Yıl | Cotton history satırı | Aynı referans gününde okunabilir eğri |
|---|---:|---:|
| 2016 | 250 | 0 |
| 2017 | 251 | 0 |
| 2018 | 251 | 0 |
| 2019 | 252 | 0 |
| 2020 | 253 | 246 |
| 2021 | 252 | 251 |
| 2022 | 251 | 250 |
| 2023 | 251 | 250 |

Bu sayılar mevcut history ile **referans-tarihi eşleşmesidir**, kabul edilmiş
forecast origin veya karar anındaki kullanılabilirlik değildir. 2016–2019'a
eğri doldurulamaz; dört yıllık kaynağı sekiz bağımsız yıl gibi sunamayız.
Eski 2.006-origin /6-of-8-yıl kapısını geçilmiş veya değiştirilmiş saymayız.
997 günlük rapor/9.970 fiyat, aynı sayıda bağımsız bilgi olayı değildir.

**VERIFIED:** 998 metadata kaydında doğrulanmış UTC `published_at` ve timezone
sayısı **0**, `first_version_reviewed=false`. 13 kayıtta görüntülenen yerel
yayın günü referans tarihinden sonra. Örneğin 2020-10-21 belgesi
2020-10-29 09:59:41 görüntüleniyor. Bu tanık ilk yayının gerçekten geç
olduğunu kanıtlamaz; ilk sürüm yerine kullanılamaz. Eski AMS exploratory
kodunun `max(reference_day, displayed_day)` koruması vardı; bu bulgu eski
deneyde doğrulanmış leakage diye sunulmaz. Sabit bir günlük gecikme eklemek
tarihsel clock/vintage boşluğunu kanıtla kapatmaz.

İlk iki vade arası **665 günde iki, 332 günde üç ay**: “next month” diye
bir aylık vade farkı varsaymak hatalı olur. Yakın kontrat/aynı sezon ve
vade kimliği korunmalı; otomatik sabit yıllıklaştırma yapılmamalı.
996 ilk fiyat Close ile `1e-5` toleransta eşleşiyor. Eski 2020-04-24 fiyat
uyuşmazlığı ve 2021-10-20 okunamayan tarih korunuyor. 2020-01-17 metadata
eksiği doldurulmadı. 54 dosyada tablonun dışındaki UTF-8 olmayan karakterler
var; tarih/futures tablo alanlarının tamamı strict ASCII olarak ayrıca
okundu. Bunun fiyat bozulması olduğu gösterilmedi; orijinal byte'lar değişmedi.

Fiyat farkları burada raporun kendi sayısal biriminde envanterlenir.
Close eşleşmesi tüm kontratlar için resmi settlement/teslim saati veya
uygulanabilir fiyat sözleşmesi değildir. **Model-eligible satır: 0.**

Gelecek vade adı gelecek gerçekleşmiş fiyat değildir: 2023-12-29 tablosundaki
Mar-24, o raporda fiyatlanan 2024 teslim kontratıdır. Bilginin tarihsel
kullanılabilirliği ayrıca kanıtlanmalıdır. İleride bir katkı bulunursa
CT=F'nin mekanik kontrat değişimlerini tahmin etmekle aynı kontratın fiyat
hareketini tahmin etmek ayrılmalıdır; gelecekteki roll kimliği özellik
yapılmaz, aynı-kontrat alt kapsamı yalnız önceden tanımlı ikincil tanı olur.

## Üç adayın elemesi; model seçimi değil

| Soru / mekanizma hipotezi | Eldeki bilgi ve önceki kapsam | Kabul / aynı-origin kontrol / durma |
|---|---|---|
| Gerçek vadeler arası fark: piyasanın aynı anda fiyatladığı farklı teslim dönemleri ek bilgi olabilir | 997 gerçek tablo, 996 farklı fiyat vektörü; eski spot basis bunu test etmedi. Clock/ilk vintage açık, 2016–2019 yok; mevcut veri maliyeti 0 | **Öncelikli kaynak adayı, eğitim GO değil.** Kaynak tarihi/vade/birim/değer sürümü ve kullanılabilirlik bağlanmadan eğitime girmez. Sayısal kol ve age/missingness placebo aynı origin/hedefte; eksik tarihler sessizce atılmaz. |
| USDA “sürprizi”: raporun piyasa beklentisinden sapması | WASDE/NASS revision'ları var; incelenen kaynak ve sicilde bağımsız beklenti arşivi belirlenemedi. Revision, surprise değildir | **Şimdi seçilmedi.** Beklenti bulunmadan yeni özellik/fit yok; eski WASDE/Texas tarifleri büyütülmez. Kaynak kabul edilmiş olursa aynı rapor/clock/eksiklik kontrolü gerekir. |
| Rejime göre koşullu katkı | `cotton_volatility_regime_20_60` zaten özellik; `diagnostics.residual_report` training quantile'larıyla price/volatility gruplarını zaten raporluyor | **Yeni ekonomik kural henüz yok.** Bu, tüm koşullu modellerin test edildiği anlamına gelmez. Eski OOS'tan kazanan rejim/eşik seçerek yeni hipotez ilan etmek yok; önceden tanımlanmış mekanizma ve ortak tüm-origin değerlendirme gerekir. |

`history check` curve/contract_curve/term_spread/surprise için eşleşme bulmadı;
AMS için 7 çalışma/14 fit tarifi, regime için 94 çalışma/498 tarif ilişkilidir.
İkinci sayı, isim eşleşmesi nedeniyle bütün koşullu hipotezlerin sınandığı
anlamına gelmez. Sonraki tam tarifte `history check --proposal` zorunludur.

## Teşhis ve sonraki tek bilimsel karar

**STRONGLY SUPPORTED:** yanlış loss fazla hatanın bir bölümünü yaratmış,
fakat aynı price_delta ve 24 özellikte doğru MAE loss'u da Naive'den kötü.
Bu testte shrinkage yok; küçük ham fark yalnız birkaç büyük hataya bağlı değil.
“Yalnız loss / yalnız küçültme / bütün Close bozuk” ana açıklamaları yeterli değil.

**ROOT CAUSE NOT YET IDENTIFIABLE:** kullanılan bilgide pratik koşullu etkinin
çok küçük olması ile mevcut temsil/tarifin onu çıkaramaması ayrışmadı.
Sentetik geri kazanım yalnız enjekte edilen etkiyi, MAE kontrolü yalnız sabit
eski iteration/parametreleri sınadı. Gerçek curve verisinin bulunması onun
tahmin becerisi olduğu veya bütün eski başarısızlığı açıkladığı değildir.

Kapasite karşı kanıtı, mevcut [loss sonuç arşivindeki](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/paired-price-loss-result-20261010)
48 makbuzdan: piyasa ağaç sayıları min **1**, medyan **40,5**, max **208**;
2019 üç seed'de **1/2/3**. Bilinen-sinyal kontrolleri farklı depth8/eta0,15
parametreleri ve **250** ağaçla, ezber kontrolü 700 ağaçla yapıldı. Bu motor/
loss kontrolüdür; her sabit piyasa tarifinin küçük etkiyi geri kazandığı
kanıtı değildir. Öte yandan 2022'nin 131/166/208 ağaçlı kolu da Naive'den
kötü; yalnız az ağaç bütün başarısızlığı açıklamıyor. Grid açma gerekçesi yok.

**Tek sonraki iş:** mevcut gerçek eğri için kaynak/clock/cohort kabul kararını
ve yalnız bu alanın ek katkısını ayıran **tek eşleştirilmiş ablation ön kayıt
önerisini** hazırlamak. Bu not ön kayıt veya eğitim izni değildir. Kontrol
mevcut Cotton bilgisi + eğriyle aynı age/missingness; müdahale tek sabit
gerçek-vade farkı. Model, loss, seçim ve hedef ortak kalır; esas soru T+1
paired fiyat-MAE katkısıdır, Naive ayrıca raporlanır. Tam tarif/fit bütçesi
kabul kararı olmadan kilitlenemez. Clock kanıtı yoksa yalnız açık varsayımlı
duyarlılık seçeneği ayrı karar gerektirir; verified PIT gibi sunulmaz.

Önerinin en küçük sayısal müdahalesi `log(F_second / F_first)`; vade ay farkı
iki kolda ortak takvim bilgisidir. On fiyatı on yeni özellik/gride çevirmek
yok. Önceden kullanılan Ridge `alpha=1`/seed42/scaled-log T+1 yolu adaydır;
training-only preprocessing, label maturity ve 3×63 geçmiş seçim korunur.
Bu bir tamamlanmış tarif değildir: kabul edilen timestamp, kaynak kapsamı
ve donmuş ortak origin'ler olmadan fit sayısı veya güç iddiası üretilmez.

Pozitif fark, eksik bilgi alanını destekler; negatif fark bu tek temsilin
pratik etkisini sınırlar, tüm bilgi yokluğunu ispatlamaz. Aralık ayıramıyorsa
belirsiz kalır; grid açılmaz. Kaynak kapısı geçmezse **fit 0**, bu tarife
geçilmez. Yeni ücretli veri, FAS kabulü veya gece görevi değişikliği yok.

## Kanıt paketinin sınırı

[Ek arşiv](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/information-admission-20261010)
1.318 üye /5.555.674 byte; SHA256
`2ce54071c9677d9ec91475df535c1bf8282208de08d0a3263b534b9465a20ff8`.
[Üye envanteri kimliği](../research/evidence/information-admission-release-20261010.json).
Mevcut 1.000 ham girdi, dondurulmuş 247 ML dosyası, inceleme anındaki 156
çalışma sicili ve private yerel yol kullanmayan okuyucu korunur. Yeni kaynak
indirmesi değildir; arşivleme tarihsel erişimi kanıtlamaz. ML kaynağı değişmedi.
Yerel safe restore, 997 tablo ve karantina/input JSON byte replay'i geçti;
GitHub'dan yeni indirme, 1.318 güvenli üye/997 bağımsız tablo ve aynı quote/
input byte replay'i de geçti. [Teslimat makbuzu](../research/evidence/information-admission-delivery-20261010.json).

Checksum/safe-member doğrulamasından sonra mevcut CPU Python ile:

```bash
python verify.py --repository repository --data-root data --output fresh-verification --base-commit e64b72c9d281c233d62f517e6073e3fa510e2c39
```

Komut eğitim yapmaz; yeni output ister. Sonuç bağımsız tarihsel holdout veya
verified-publication paketine dönüşmez. Sicilde eski 156 çalışma/508 fit
tarifi ve 60 kanıt korunur; yalnız bu sıfır-fit inceleme eklenir, trials değişmez.
