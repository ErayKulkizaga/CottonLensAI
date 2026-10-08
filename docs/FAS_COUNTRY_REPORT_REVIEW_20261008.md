# FAS ülke raporu mutabakatı — 8 Ekim 2026

**Karar: eğitim kabulü açılmadı.** Mevcut iki rapor haftasında dört ülkenin
24 stok kontrolünden 22'si sabit ±50 balya sınırında eşleşti. 28 Mayıs 2020
Çin/Pakistan birikimli ihracatında sınır dışı iki küçük fark bulundu. Ulusal
toplam eşitliği ülke alanlarının eşitliğini garanti etmiyor. Bu bir piyasa
deneyi veya FAS'ta sinyal yokluğu sonucu değildir; **sıfır yeni fit**.

## Doğrulanan kapsam

- Çin 5700, Vietnam 5520, Türkiye 4890, Pakistan 5350: checksum bağlı mevcut
  [Census kod listesi](https://www.census.gov/foreign-trade/schedules/c/countrycodes.html)
  ve [USDA talimatlarının](https://www.reginfo.gov/public/do/DownloadDocument?objectID=135773101)
  web metnindeki dört eşleştirme. USDA belgesinin özgün PDF byte'ları alınamadı;
  web gözlemi dosya checksum kanıtı veya bütün tarihsel ülke kataloğu sayılmaz.
- 28 Mayıs ve 4 Haziran 2020 Upland tabloları: outstanding, accumulated exports,
  next-MY outstanding. CURRENT kolonları, YR AGO/Pima satırlarından ayrıldı.
  Her haftada 4 ülke × 3 alan. İki ulusal toplam kontrolü de geçti.
- Kullanılan 2020 API snapshot hash'i, önceki 15-kaynak alan denetimindeki
  2020 hash'iyle aynı: `9973f506fa67950687cb3f945c36cf601ed3cf8c49420ee2ada45bfc52ad3c83`.
  Bulgular farklı bir API indirmesine dayanmaz.
- 4 Haziran için USDAFAS GovDelivery kopyasında aynı 12 stok değeri eşleşti.
  Bu üçüncü bağımsız hafta veya tüm belgenin aynı sürüm olduğunu göstermez.
- Commitment, outstanding + accumulated olarak API'de doğrulandı;
  raporda ayrıca basılmış bağımsız bir alan gibi sayılmadı.

| Hafta / ülke / alan | API, balya | PDF, balya | API − PDF | Kontrol |
|---|---:|---:|---:|---|
| 2020-05-28 / Çin / accumulatedExports | 1.478.645 | 1.478.700 | −55 | Başarısız |
| 2020-05-28 / Pakistan / accumulatedExports | 1.760.947 | 1.761.000 | −53 | Başarısız |

PDF'nin 34. sayfası görsel olarak ve bağımsız pdfplumber 0.11.9 ile kontrol
edildi; pypdf 6.10.0 ile aynı basılı değerler bulundu. Farkın revizyon, rapor
hesaplaması veya başka bir nedenden geldiği **belirlenmedi**. Sınır yalnız
5/3 balya aşılmış olsa da tolerans sonucu geçirmek için genişletilmedi.
Bu büyüklükteki farkın önceki tahmin başarısızlığını açıkladığı gösterilmedi.
Hiçbir API/PDF değeri değiştirilmedi veya “düzeltilmiş veri” diye yeniden yazılmadı.

## Tamamlanmayan kontroller ve tek sonraki iş

Haftalık exports/netSales basılı ülke değerleri doğrulanmadı; ardışık
accumulated farkından haftalık sevkiyat türetilmedi. İki hafta, 752 haftanın
veya 64 kodun tarihsel doğrulaması değildir. Yayın saati, ilk sürüm ve belirli
byte sürümünün tarihsel erişimi hâlâ bilinmiyor. 52 arşiv kaydının createdTime
alanı placeholder; PDF CreationDate, URL tarihi, HTTP Last-Modified ve basılı
embargo tek başına fiilî teslim zamanı olarak kullanılmadı.

**Sonraki tek iş:** 28 Mayıs 2020 için belirli özgün yayın sürümünü bulup
bu iki accumulated değerini onunla karşılaştırmak; bulunabilirse aynı sürüme
bağlı tarihsel erişim kaydını saklamak. Böylece API/arşiv sürüm farkı ile basılı
raporun hesaplama/yuvarlama farkı ayrılır. Kanıt bulunamazsa uyuşmazlık ve
erişim bilinmeyeni açık kalır; otomatik eğitim veya tolerans değişikliği yoktur.
Piyasa katkısı sınaması yalnız kaynak kabulü yeterliyse ayrı ön kayıtla açılır.

## Yeniden üretim ve koruma

[Küçük kanıt](../research/evidence/fas-country-report-verification-20261008.json),
[tam mutabakat kaydı](../research/evidence/fas-country-report-review-20261008.json)
ve [checksum bağlı Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fas-country-report-20261008)
girdi, kod, okuyucu ve her alanın farkını saklar. Release'teki README çevrimdışı
komutu ve restore kontrolünü içerir. `--country-reference` verilmezse eski
ülkesiz inceleme korunur. Ülke uyuşmazlığında kayıt korunur ve CLI exit **2**
verir; eşleşme de model kabulü anlamına gelmez.

10 eski girdi, 2 yeni referans dosyası ve 21 eski Git kanıt dosyasının hash'leri
eşit; 133 önceki sicil kaydı korunur. Tarif sicili değişmedi: 435 tarif /
25.000 tamamlanmış fit makbuzu. Ham kaynaklar, geçmiş tahminler ve canlı artifact
bu çalışmada değiştirilmedi. Yeni kayıt piyasa sorusu için **INCONCLUSIVE**;
sayısal uyuşmazlık doğrudan doğrulanmış ayrı bir bulgudur.

Yerel doğrulama: 739 ML testi geçti, 3 test atlandı; Ruff ve sicil doğrulaması
geçti. Release'in 19 üyesi ve 215 kaynak dosyası ayrı dizinlere checksum ile
geri kuruldu; çevrimdışı tekrar aynı kayıt gövdesini ve beklenen exit 2
sonucunu üretti. GPU ve gerçek piyasa eğitimi çalıştırılmadı.

PR CI'sinde mevcut log testi, asenkron Drive kopyası henüz bitmeden okuma
yaparak bir kez başarısız oldu; aynı ilk commit'in diğer iki CI çalışması
geçti. Test artık yerel kaydı hemen, Drive kopyasını en fazla 5 saniyede
doğruluyor; supervision/eğitim kodu değişmedi. Kanıt Release'inin kaynak
snapshot'ı incelemenin yürütüldüğü `fb4f1a71343e4d95f961f9d091d502a7ff58effe`
commit'ine aittir; sonraki yalnız-test düzeltmesi donmuş kanıtı değiştirmez.
