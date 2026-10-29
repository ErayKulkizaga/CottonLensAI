# Risk bilgi tanıları: üç sıfır-fit kontrol ve takvim erişimi

Bu kontroller model/tahmin üstünlüğü değildir. Aynı kaydedilmiş CT T+1
kare-getiri/EWMA0,94 ve2016–2023/2.006ordinal origin kullanıldı.
Her tanının kuralı hesap öncesi ayrı contract'a yazıldı. Yıl içi covariance,
full-ordinal20/60bootstrap,10.000/seed42 ve top1% EWMA-loss duyarlılığı;
centering her bootstrap örneğinde yeniden hesaplandı. Araştırma geçmişi
yeniden kullanılmıştır; çoklu araştırma kararlarına göre düzeltilmemiştir.

| Tanı | Covariance | CI20 | CI60 | Pozitif yıl | Trim |
|---|---:|---|---|---:|---:|
| Log range/EWMA vs relatif EWMA hatası | −0,92450 | [−2,41487;0,76345] | [−2,43848;0,76667] | 2/8 | −0,29700 |
| Bugünkü high==low işareti | +0,04799 | [−0,02800;0,11128] | [−0,02551;0,11192] | 6/8 | +0,01695 |
| Hedef günde dated WASDE dokümanı | +0,01151 | [−0,01312;0,03473] | [−0,01082;0,03295] | 4/8 | +0,01537 |

Üçü de ön kayıtlı pozitif screen'i geçmedi. Bunlardan yeni range/flag/calendar
learner, grid veya GPU işi açılmaz. Nonsignificance bilgi yokluğunun ispatı değildir.

Range tanısında H/L/open/close containment koşulu1.738origin bırakır;
settlement'in aralık dışında olması otomatik veri bozulması sayılmaz.
İkili flag yalnız finite high>=low>0 ister;2.006origin korunur,248flat bar.
Bu farklı scope açıkça kaydedildi; ilk maske/çıktı değiştirilmedi.

WASDE90dated narrative kayıtlarından olay gününü sonradan etiketler;
**uses_ex_post_report_day=true**. Canlı feature değildir. Beş corpus-eksik
ayın bütün hedef günleri unknown;1.902origin/90dated event kalır.
2019-01 arşivde yayımlanmamış olarak ayrı kayıttır. Feature timing/vintage
kanıtı olmadan bu event bit'i modele giremez. T+5 numerical/text kaynak
kontrolleri, önceden bilinen T+1 olay takvimiyle eşdeğer değildir.

## Ayrı bounded belge erişim sonucu

USDA/FAS WAP2015-12,2020-01,2023-05PDF'lerinde sonraki yayın tarihi
metinde bulunuyor:2016-01-12,2020-02-11,2023-06-09. İlk örneğin8.sayfası
Poppler ile görsel olarak doğrulandı. Current download receipt/SHA/bytes
saklandı; ilk5MB sınırında2020dosyası5.584.225bytes nedeniyle durdu,
ayrı20MB bounded kontrolle üç örnek tamamlandı. Eski hata gizlenmedi.
ÜçPDF toplam22.139.985bytes;0TL/0fit. Dosyalar yalnız yerelde, Drive'a yazılmadı.

Bu WAP gelecekteki tarih duyurusudur; WASDE ile takvim eşliği, tam arşiv,
ilk-vintage, exact UTC veya predictive skill ispatı değildir. **after12:00ET**
exclusive lower bound'dur. Önceki raporun printed tarihi gerçek first-publication
kanıtı sayılmaz. Screen başarısızken tüm96ay için PDF arşivi indirilmedi.

Ekonomik gerekçe:[USDA ERS WASDE etkisi araştırması](https://www.ers.usda.gov/amber-waves/2012/june/wasde).
Bu eski araştırma mevcut CottonLens sonuçlarının yerine geçmez.

Kanıt yolları:`output/risk-range-moment-20261013/`,
`output/risk-wasde-event-moment-20261013/`,
`output/wap-calendar-preflight-20261013/receipt-v2.json`.
Yeni verifier math unit testleri4passed/Ruff; eski model, fiyat, kaynak
makbuzları ve live126-origin kilidi değiştirilmedi.

## Kalıcı teslimat

Private GitHub `risk-information-preflight-20261013` Release taze indirildi;
25 üye checksum ile geri kuruldu. Bağımsız doğrulama6.018Decimal satır,
6covariance aralığı ve3PDF hashini yeniden denetledi; verification JSON
byte eşit. Yeni fit0, eski196aşama ve tüm fit tarifleri korundu.
Kanıt: `research/evidence/risk-information-preflight-20261013.json`.
