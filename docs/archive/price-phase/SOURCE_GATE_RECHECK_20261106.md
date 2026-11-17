# Gerçek kaynak paketi yeniden denetlendi

**Named-risk piyasa eğitimi kabul edilmedi.** Sıfır yeni fit ile private GitHub'dan
401 dosya hash doğrulanarak geri kuruldu. Arşivdeki kaynak kodla11 çıktı byte eşit
yeniden üretildi. Bağımsız dört günlük PDF kontrolü120 fiyat scalar'ını; feature
kontrolü2.829 Decimal scalar'ı doğruladı. Önceki139 evaluation HAR farkı ve
106train/28test işi etkisi yeniden doğrulandı.

[Sayısal makbuz](../../../research/evidence/source-gate-review-20261106.json), çalışma
payload'larının ve güncel iki kaynak dosyasının hash'lerini içerir. Büyük dosyalar
yerel output/data-gpu-audit altında. Yeni paket artık
[private Release'te](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/source-gate-l4-audit-20261106):
224dosya taze restore edildi;9katı-kaynak çıktısı byte eşit replay edildi.
Parent401dosyanın hash'i ayrıca kontrol edildi; eski parent arşivi tekrar
yüklenmedi. L4sonucu ve bağımsız NumPy/Adam verifier raporu da taze restore
üzerinden byte eşit üretildi. [Teslim makbuzu](../../../research/evidence/source-gate-delivery-20261106.json).

## İlk katı kontrolün etkisi (korunan önceki sürüm)

| Koşullu değerlendirme kapsamı | Eski | Katı |
|---|---:|---:|
| Hedef fiyat çifti |1955|1955|
| HAR hazır |1794|1647|
| EWMA hazır |667|0|
| Bütün girdiler hazır |662|0|
| Potansiyel bilinmeyen EWMA hücresi üst sınırı |2053|40450|

174işin tamamında hazır olmayan train/test bağımlılığı var. İkinci bağımsız
set/bisect hesaplayıcı aynı sonuca ulaştı. Bu fark, kayıp kaynak tablosunun
`complete=True` sayılmasının ve kısmi tablodan ay/sıra yoluyla yokluk çıkarımının
kaldırılmasından gelir. 40.450gerçek fiyat hatası bulundu anlamına gelmez.
Muhafazakâr kapsam hesabında bilinmeyen olası güncellemeler EWMA'da taşınır;
sonradan23temiz bar HAR'ı toparlasa da EWMA başlangıcı sessizce sıfırlanamaz.

Bilinen bozuk dönem dışındaki annual girdiler hâlâ koşullu kullanılıyor;
0hazır sonucu tüm veri evreninin mutlak kullanılmazlığı değil, bu kilitli
named-risk kapsamının kabul edilememesidir. CT=F fiyat-only çalışması hakkında
bu sonuçtan otomatik hüküm çıkarılamaz. Fiyat doğruluğu ve first-vintage/saat
kanıtı ayrı kalır; G1 geçilmedi.

## Yeniden üretim sınırları

İlk replay hostPython3.13'te Parquet engine eksikliğinde durdu. İkinci deneme
Python3.12CPU ortamıyla hesapları doğruladı; PDF okuyucu0.11.7 yerine arşivdeki
0.11.9 kullanılmadığından rapor metadata/hash'i farklıydı. Üçüncü deneme0.11.9
ile11çıktıyı byte eşit üretti. Başarısız girişimler korunuyor; veri, tolerans veya
eski rapor değiştirilmedi. PDF paketleri yalnız output altındaki izole klasöre
kuruldu. Eğitim bağımlılıkları/üretim backend'i değiştirilmedi.

Sıradaki veri işi: erken tam kontrat tanıkları ve listeleme kapsamını bulmak;
gerekirse kabul testiyle ücretli kaynak teklifi. Mevcut makbuzlarda kanıtlanmamış
tarihsel yayın zamanını varsayarak kabul vermek çözüm değildir.

## Erken katalog kapsamı kontrolü

6 Kasım'da resmi ESMIS HTML tarih filtreleri yeniden indirildi: tam günlük
`daily-spot-quotations` seçenekleri2018-12–2025-09; `excerpts` seçenekleri
2018-11–2025-07. Sabit örnek2018-08-01 her iki filtrenin dışında. Bu yalnız
gözlenen katalog arayüzünün kapsamıdır; başka arşivde belge bulunmadığını veya
geçmişte yayınlanmadığını göstermez. Özet rapor tam on-kontrat tablosunun yerine
geçmez; filtre seçeneği de tarihsel first-publication kanıtı değildir.

[Makbuz](../../../research/evidence/early-catalog-review-20261106.json) ham HTML hash'lerini
ve gözlenen bütün ay seçeneklerini korur. Aynı private Release'e ayrı
`early-catalog-review-20261106.zip` asset'i eklendi; taze indirme sonrası üç üyenin
tam byte eşitliği ve HTML hash'leri doğrulandı. Eski paketlerin üzerine yazılmadı.
G1 kapalı; yeni piyasa fit'i yok.

## Listing kapsamı kontrolüyle güncel durum

Tam on-quote rapor ile bütün listeli kontrat evreni ayrıldı. Artık report
complete=True tek başına rapor dışındaki kontratı ABSENT yapmıyor; ayrı
absence_coverage_verified gerekir. Mevcut source constructor'ları bunu otomatik
vermez. Bağımsız verifier da rapor/annual sayısından yokluk çıkarmıyor.

Aynı gerçek kaynak/joblardan sıfır fit ile yapılan yeni hesap:99.523potansiyel
bilinmeyen hücre,3.108kaynak tarihi; önceki40.450hesabı arşivde korunur. Bunlar
fiyat hata sayısı değildir.1955hedef çifti/1647HAR hazır/0EWMA hazır/0tüm-girdi
hazır değişmedi;174işin train ve test bağımlılıkları hâlâ eksik. Bağımsız set/bisect
verifier aynı99.523hücreyi ve bütün2824origin/174işi doğruladı.

Resmi ICE canlı expiry HTML/CSV'si15kontrat gösteriyor; eski58kontrat için
FTD/LTD kanıtı olarak kullanılmadı. Güncel sayfa geçmişi doğrulamaz, fakat
on-satırlı raporun evrensel listing kapsamına eşitlenemeyeceğini açıklaştırır.
Kaynak talebi güncel58kontrat matrisiyle lifecycle-first yapılır; fiyatlar veya
yokluklar uydurulmaz. Koşullu annual ve PIT/vintage açıkları hâlâ kapanmadı.
