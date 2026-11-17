# Tarihli haftalık kontrat kanıtı — 29 Ekim 2026

**VERIFIED:** Eski AMS annual kontrat kimliği hatasının gerçek HAR girdilerine
etkisi genişletilmiş kanıtta da görülüyor: 138 değerlendirme tarihi,
174 ortak işin 106 train ve 27 test girdisi etkilenmiş. Önceki kanıtta
bunlar 110/96/20 idi. Bu sayılar performans kaybı ölçümü değildir.
**Piyasa başarısı gösterilmedi; yeni fit 0.**

## Yapılan iş

Sabit 33 aday yayın tarihinin tek seferlik manuel taraması tamamlandı.
19 USDA AMS haftalık PDF'inin basılı tarihi beklenen tarihle eşleşti;
33.427.755 byte korundu. Dosyalar Barchart'ın S3 arşivindeki USDA kopyalarıdır;
ayrı bir bağımsız piyasa fiyat sağlayıcısı veya first-vintage kanıtı değildir.
14 aday doğrulanmadı. 2018-09-21 dosyası 5MB tek-dosya sınırını aştı;
diğer erişim hataları da saklandı. Otomatik/saatlik collector açılmadı.

İlk sayfadaki raster Futures Settlement tablolarının altı başlığı ve günlük
hücreleri görsel olarak okunup açık transkripsiyonla kaydedildi. OCR çıktısı
veya annual fiyatı kullanılarak eksik hücre tahmin edilmedi. 19 tablo,
95 tarih satırı ve 562 görünen fiyat hücresi vardır. Bir satır açık tatildir;
iki satırın ilk kontrat hücresi açıkça `-` işaretidir. Bu işaretler korunur.

| Ölçüm | Son doğrulanmış değer |
|---|---:|
| Kabul edilen kısmi fiyat/kimlik günleri | 91 |
| Eski çözümsüz 97 gün içinde yeni doğrulanan prefix | 73 |
| Aynı iki şüpheli annual blokta kalan çözümsüz gün | 24 |
| Eski daily katmana eklenen kısmi gün | 76 |
| Birleşik katmandaki gün | 1.331 |
| Tam 23 Cotton gözleminden kurulabilen HAR penceresi | 917 |
| Bağımsız Decimal HAR scalar kontrolü | 2.751 |
| Gerçek eski HAR değeri farklı değerlendirme tarihi | 138 |

Weekly tablonun beş/altı kontratı, daily kaynağın on kontratıyla aynı kapsam
değildir. Diğer dört/beş kontrat uydurulmaz. Birleşimde mevcut on-kontrat
daily günü altı-kontrat veriyle daraltılmaz; uyuşmazlıkta birleşim reddedilir.
Özgün 2.006 ordinal slot ve eski model/tahmin/veri dosyaları korunur.

## Somut hatalar ve korumalar

- 2018-09-05 weekly tablosu annual'dan orta kolonlarda ayrılıyor;
  2018-11-14'te beş kolon ayrılıyor. Görsel okuma bu farklılıkları doğrular;
  fiyatlar daha makul görünsün diye değiştirilmez. İki gün kabul edilmedi.
- 2018-11-26 weekly ile annual uyuşmasına rağmen aynı tarihli MP_CN001
  farklı fiyat veriyor: ilk iki kontrat weekly/annual'da 77,40/78,58,
  daily'de 76,75/77,93. Bu bir fiyat alanı/vintage veya kaynak çatışmasıdır;
  hangi tarafın doğru olduğu kanıtlanmadı. Gün çözümsüz kaldı.
- Önceki kabul edilmiş overlay'e bakmak reddedilmiş günlük raporun çelişkisini
  gizleyebilir. Yeni kontrol **reddedilmiş ham günlük tanıkları da** açar.
  İlk geçici rapor bu kontrolle değiştirildi; eski geçici dosya silinmedi.
- 2018-08-15 annual yedinci kontratta 76,843 içeriyor. İlk altı hücre
  weekly ile eşleşiyor. Uzak bozuk hücreyi silip sırayı sıkıştırmadan,
  yalnız doğrulanan ilk altı fiyat ayrı katmanda korunur. Bozuk hücre düzelmiş
  veya tüm on kontrat doğrulanmış sayılmaz.

Geçici v1/v2 raporları ve düzeltme öncesi kaynakları ayrı `superseded/`
alanında tutulur. Son kanıt `execution/` içindeki recovery-v3'tür.

## Kanıt ve sınır

`research.ams_weekly_recovery` yalnız yerel kaynak karşılaştırır;
`research.verify_ams_weekly_recovery` kaynak hash'lerini, basılı PDF tarihini,
ham günlük fiyatları ve 562 hücreyi bağımsız Decimal yoluyla kontrol eder.
`research.ams_weekly_merge` eski tam günlük fiyatları korur. Eski bağımsız
feature verifier değiştirilmeden 2.751 scalar'ı tekrar kurdu.
22 ilgili test ve Ruff geçti. UI/backend/live model değişmedi.

Kaynak kimliği `06bd0ad326e6667b2440d2d7d1790d6c7d830cb9e7ca3dbe2ed361aacb6437e7`.
Private [Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ams-weekly-contract-recovery-20261029)
ham PDF/görselleri, makbuzları, transkripsiyonu, önceki ham tanıkları ve tam
yeniden kurma kaynağını taşır. GitHub'dan taze indirilen1.700dosya hash ile
geri kuruldu. Donmuş kaynakla iki rapor, iki CSV, iki overlay ve iki bağımsız
makbuz byte eşit tekrar üretildi; ağ erişimi ve fit kapalıydı. Arşiv SHA
`e9cca58813d2e668f469117ea868342093c78e10ef08a7458a70b69dfd70f3e9`.

Bu katman **training-admitted değildir**. Basılı haftalık yayın daha sonraki
bir tarihte olabilir; daily karar anına bilgi taşındığı varsayılamaz.
Gerçek tarihsel UTC ve ilk sürüm hâlâ doğrulanmadı. Düzeltme sonrası HAR/EWMA
tahminleri veya yeni başarı skoru henüz üretilmedi. CT ve önceki yalnız 2023
named deneyleri bu bulguyla otomatik geçersizleşmez.

## Tek sonraki bilimsel iş

Özgün 2.006 slot ve 174 ortak iş üzerinde **gerekli hücre kurulabilirliği**:
her origin'in hedef kontratı, 22 geçmiş eşlenmiş getirisi ve EWMA'nın bütün
başlangıç/geçmiş durumu için hangi kanıtın eksik olduğunu çıkar. Yalnız bu
hesabın gerekli bulduğu tarihler/kontratlar tamamlanır. Kalan 24 günün
tamamı aynı model girdisini engelliyormuş gibi davranılmaz; sessiz origin
kesişimi, ilk altıya bakarak uzak EWMA geçmişini varsayma veya yeni grid yoktur.
Tam kurulabilirlik ve frozen kaynak teslimatı olmadan yeni named fit açılmaz.
