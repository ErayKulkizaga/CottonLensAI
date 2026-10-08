# FAS: tarihsel kod tanığı ve özgün hücre erişim sınırı

**Sonuç:** dört ülke kodu için tarihsel tanık bulundu. Çin'in 2020-05-28
accumulatedExports özgün tam-balya hücresi bulunmadı; yuvarlama ile sayısal
sürüm farkı hâlâ ayrılamıyor. **Sıfır fit, kaynak kabulü yok.**

## Yeni kanıt

Eski resmî ESR sorgu adresinin Wayback CDX indeksinde 2020 için yedi kayıt
var. Hedef hafta öncesindeki 17 Nisan ve sonrasındaki 1 Temmuz incelendi.
İndirilen HTML byte'ları CDX'in SHA1/Base32 payload digest'leriyle eşleşiyor;
ayrıca SHA256 ile donduruldu. CDX zamanı arşiv yakalama tanığıdır, USDA yayın
veya ilk erişim saati değildir.

| Arşiv formundaki tam ülke etiketi | Ülke kodu | Nisan / Temmuz literal option |
|---|---:|---|
| CHINA, PEOPLES REPUBLIC OF | 5700 | `7:5700` / `7:5700` |
| VIETNAM | 5520 | `9:5520` / `9:5520` |
| TURKEY | 4890 | `2:4890` / `2:4890` |
| PAKISTAN | 5350 | `9:5350` / `9:5350` |

`1404 = All Upland Cotton` iki formda var. Öneklerin anlamı kullanılmaz;
balya birimi bu formlarla doğrulanmadı. **VERIFIED:** iki arşiv payload'ında
bu dört etiket/kod eşleşiyor. **NOT SUPPORTED:** aradaki kesintisiz geçerlilik,
64 kodun veya 2016–2023'ün tamamının tarihsel kabulü. Önceki “yalnız güncel
dört kod tanığı” sınırı bu dar kapsam için ilerledi; eski kayıtlar değiştirilmedi.

## Özgün hücreyi vermeyen yollar

Arşiv kaydı sorgu formudur; çalıştırılmış ülke/hafta sonucunu içermez.
Eski formu bugün POST etmek 2020 sayısal sürümünü geri getirmez; yapılmadı.

[Resmî ESRQS kılavuzunun](https://apps.fas.usda.gov/esrqs/assets/ESRQS_UserManual_PublicUser.pdf)
20–21. sayfalarına göre Weekly Export Adjustments, seçilen haftada bildirilen
eski/yeni miktarları ve özgün hafta tarihini gösterir; bu tam vintage değişim
günlüğü veya birikimli ihracat stokunun özgün değeri değildir. İlgili sayfa
görsel olarak da denetlendi.

4 Haziran 2020 için resmî rapor görünümü, HTTP 200 ve doğru tarih parametresiyle
yalnız kolon başlıkları döndürdü. Canlı görünüm de başlık dışında satır
göstermedi. **Boş görünümün nedeni/kapsamı bilinmiyor; “hiç revizyon yok”
çıkarılmaz.** Bir haftaya bakmak daha sonraki tüm değişimleri zaten dışlayamaz.

28 Mayıs **gözlem dönemi sonudur**, kanıtlanmış yayın tarihi değildir.
Bugünkü API'nin 1.478.645 değeri özgün yayın değerinin yerine geçirilmeyecek.

## Karar ve tek eksik girdi

Bu sınırlı erişim denetimi tamamlandı; aynı PDF, aynı form ve aynı boş sorgu
yeniden araştırma işi olarak açılmaz. Sicilde `fas-historical-reference-witness-v1`
ve `as_issued_cell_access = BLOCKED_EXTERNAL_EVIDENCE` olarak kayıtlıdır.
Yeni kaynak gelirse kimliğiyle kontrol tekrarlanabilir. Otomatik eğitim yok.

Eksik girdi: **ilk yayın sürümüne bağlı Çin (5700), All Upland Cotton (1404),
hafta sonu 2020-05-28, accumulatedExports kesin balya değeri**. Yayıncıya
iletilebilecek hazır tek-hücre talebi:

> Please provide the exact original as-issued accumulatedExports value in
> running bales for China (country 5700), All Upland Cotton (commodity 1404),
> week ending May 28, 2020. Please identify the original release/version and
> supporting file or revision record. We have a later API snapshot of
> 1,478,645 and a rounded PDF value of 1,478.7 thousand bales; we need to
> distinguish printed calculation/rounding from a numeric vintage change.

[Resmî ESRQS ana sayfası](https://apps.fas.usda.gov/esrqs/#/home), 8 Ekim
incelemesinde ESR iletişimi için `ESR@usda.gov` veriyordu. **Mesaj gönderilmedi.**
Yanıt, destekleyen dosya olmadan otomatik kabul değildir; tek hücrenin çözümü
bütün geçmişi kabul ettirmez.

## Yeniden üretim ve koruma

Yeni çevrimdışı `ml/review_fas_reference.py`, tam alan/etiket/kod eşleşmesini,
SHA256'yı, CDX payload digest'ini ve makbuz/indeks bağını kontrol eder. Eksik,
yinelenen veya yanlış kod, URL, değişmiş byte ve unsafe path sonuç kaydı
oluşturmadan reddedilir. Model/vintage/yayın kabulü vermez.

[İnceleme](../research/evidence/fas-historical-reference-review-20261008.json),
[erişim kaydı](../research/evidence/fas-as-issued-cell-access-20261008.json) ve
[Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fas-historical-reference-20261008)
ham tanıkları ve kaynak kimliğini korur. 27 eski kanıt dosyası, 135 önceki
sicil kaydı ve trials aynı; 435 tarif / 25.000 fit makbuzu değişmedi.
96 ilgili test ve Ruff geçti. Gerçek piyasa/GPU eğitimi yapılmadı.
15 arşiv üyesi ve 218 kaynak dosyası yeni dizinde checksum ile doğrulandı;
o kaynaktan çevrimdışı tekrar, dondurulmuş incelemeyle tam JSON eşitliği verdi.
