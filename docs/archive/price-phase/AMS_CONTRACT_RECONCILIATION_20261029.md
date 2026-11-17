# AMS kontrat kimliği: doğrulanmış kısmi onarım

**VERIFIED:** yıllık Futures tablosu bazı dönemlerde değişen vadelerin
fiyatlarını sabit teslim ayı başlıklarına bağlamış.26Şubat2020 günlük
MP_CN001 metninde65,47, Mayıs2020 kontratıdır; yıllık dosyada aynı fiyat
Temmuz2019 görünür. Ham fiyatlar çoğunlukla aynı; kimlikleri farklıdır.
Bu fark generic ikinci sıranın geçmişini tek kontrat geçmişi gibi kullanmaya
ve kontrat geçişinde yanlış getiri üretmeye yol açar.

## Ne gerçekten düzeltildi?

| Kontrol | Sonuç |
|---|---:|
| Ham metninden doğrulanan günlük rapor |1.262|
| On fiyatın tamamı yıllık vektörle eşleşen gün |1.255|
| Kimliği farklı, fiyatı aynı gün / kontrat hücresi |332 /3.320|
| Korunan özgün değerlendirme slotu |2.006|
| Bağımsız yeniden kurulabilen hedef çifti |1.225|
| Sayısal kare-getirisi değişen doğrulanmış çift |8|
| Tam23Cotton gözlemiyle doğrulanmış HAR penceresi |863|
| Kaydedilmiş HAR girdisi farklı değerlendirme origini |110|
|174ortak iş içinde değişmiş train / test girdili iş |96 /20|

İlk1.049 rapor mevcut hash-doğrulanmış arşivden okundu. Sabit230eksik tarih
için tek seferlik, elle başlatılmış447istek/5.245.758byte kaynak toplaması
213ek günlük raporu doğruladı;17başarısız/uyuşmayan tarih değiştirilmedi.
Otomatik görev kurulmadı, model fit'i yapılmadı. Kaynak indirme bütçesi
460istek/40MB idi. Kısmi kayıtlar ve başarısızlıklar korunur.

Bu, ikinci bir piyasa veri sağlayıcısı doğrulaması değildir: yıllık
dönüşümden önceki USDA günlük `FUTURES TODAY` başlıklarıyla tutarlılık
denetimidir. Fiyat eşliği, exchange-certified final settlement veya ilk
yayın saati/vintage sertifikası sayılmaz.

`research.ams_contract_reconciliation` fiyat vektörü ve bağımsız tarihli
kontrat başlıkları birlikte eşleşmeden onarım üretmez. Dönemlere göre yıl
tahmin edilmez. Üç fiyat uyuşmazlığı ve dört eksik yıllık gün otomatik
çözülmez. Yalnız on fiyatı kanıtlanan ayrı overlay üretilir; raw, history,
eski model, tahmin veya sicil kayıtları değiştirilmez.

## Hatanın gerçek eğitimlere etkisi

`research.ams_contract_feature_impact` her originde mevcut ikinci gerçek
kontratın önceki23Cotton tarihini ister; eksik tarih üzerinden pencereyi
sıkıştırmaz. HAR1/5/22 bu23fiyattan kurulur ve eski own-HAR alanlarıyla
karşılaştırılır. Native174işin tarih listeleri, değişen110originle
eşleştirildi. **96train/20test işinin girdisi doğrudan etkilenmiştir.**
Bu, hatanın yalnız görüntülenen kontrat adı olmadığını gösterir. Etki
sayısı alt sınırdır; doğrulanamayan pencereler doğru kabul edilmedi.
Doğrudan QLIKE performansındaki etkisi henüz ölçülmedi.

Sekiz bilinen hedef düzeltmesi mevcut tahminler/reference sabitken orijinal
1.989eşlenmiş ortalamaya eski4param seçilmiş modelde−0,00050389,
own-level seçilmiş modelde+0,00069493 katkı yapar. Bu **tam düzeltilmiş
performans skoru değildir**: diğer hedefler, bozuk özellikler ve EWMA state'i
tam düzeltilmedi. Fiyat etiketlerini değiştirmekle yetinen yeni eğitim açılmaz.

## Kanıtın sınırı ve kalan tek iş

İki şüpheli yıllık blokta97gün hâlâ çözümsüzdür(2018:84,2019:11,2020:2).
Bunların dışında2020-12-04/2021-02-09 fiyat uyuşmazlıkları da korunur.
Geçmiş teslim ayını reddetmek gerekli bir kontroldür; gelecekte görünen
yanlış teslim aylarını yakalamaya tek başına yetmez.

3Ağustos2018 tarihli USDA haftalık raporunun arşiv kopyası bulundu;
basılı tarihi ve altı kontratlı raster tablosu görsel olarak doğrulandı.
Bu yalnız erken dönem değer/kimlik kanıtının bulunabildiğini gösterir;
OCR denemesi henüz yeni bir veri kabul kuralı veya onaylanmış overlay değildir.
Haftalık raporun sonradan yayımlanması günlük origin'de erişim kanıtı sayılmaz.

**Tek sonraki bilimsel iş:** kalan dönemleri bağımsız tarihli kontrat/fiyat
kanıtıyla tamamlayıp gerçek kontrat geçmişinden HAR ve EWMA'yı kurmak.
Sonrasında kaynak kabulü, aynı takvim/minimum olgun eğitim/validation
kontrolleri geçmeden yeni fit açılmaz. Bir yıllık başlığı otomatik kaydırmak,
eski vadeleri silerek başka sıra seçmek veya model/grid büyütmek çözüm değildir.

Bu kusur, ilgili AMS full-year piyasa hükümlerini metodolojik olarak
sakatlar; önceki CT ve yalnız2023named deneylerini otomatik geçersizleştirmez.
**NOT SUPPORTED:** bütün tahmin başarısızlığını bu kusurun açıkladığı veya
onarımın EWMA'yı mutlaka geçireceği. Gerçek piyasa üstünlüğü hâlâ yoktur.
Saat/vintage ve görülmüş yılların geliştirme dönemi niteliği korunur.

## Doğrulama

İki ayrı verifier12.620günlük quote scalar'ı,1.225Decimal hedefi ve
2.589Decimal HAR scalar'ını denetledi.10yeni test ve Ruff geçti.
Tahmin veya baseline değiştiren fit yoktur. Kaynak paketinin taze GitHub
geri kurması ve sıfır-fit replay makbuzu ayrı teslimat kaydına bağlanır.
