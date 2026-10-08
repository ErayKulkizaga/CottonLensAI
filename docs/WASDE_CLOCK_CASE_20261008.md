# WASDE 11 Aralık 2018: değer sürümü ve karar saati

**VERIFIED:** 11 ve 14 Aralık arşivlerinde PDF baytları farklı; Cotton değerleri
aynı. **STRONGLY SUPPORTED:** bu olayda repost, adayın Cotton bilgisini sonradan
üretmiyor. **Henüz doğrulanmadı:** incelenen değerlerin ilk karar kesimi
`2018-12-12T00:15:00Z` öncesinde erişilebilir olduğuna ilişkin saat üst sınırı.

Bu, önceki teslimatta belirlenen **tek raporluk kontrolün tamamlanmış sonucu**.
Eğitim yapılmadı; erişim onayı verilmedi. Kanıtın bulunamaması, raporun o saatte
yayımlanmadığının kanıtı değildir. Bu aday eski tamamlanmış modellere kabul
edilmediği için saat belirsizliği, onların başarısızlığının gösterilmiş nedeni
olarak sunulamaz.

## Doğrudan yeniden kontrol

İki PDF/XML çiftinin her biri önbellekteki `passed` yerine yeniden çıkarıldı:
**196 + 196 hücre, sıfır fark**. İki XML'in normalize edilmiş Cotton kayıtları
aynı; crop year 2018 ve adayın 13 seviye/oranı da eşleşir. Önceki resmî
as-reported CSV mutabakatı ayrıca korunur. Tam PDF metni/baytları aynı denmedi.

| Alan (million 480-pound bales) | 11 Aralık | 14 Aralık | Üniversite kopyası |
|---|---:|---:|---:|
| Dünya üretimi | 118,74 | 118,74 | 118,74 |
| Dünya tüketimi | 125,63 | 125,63 | 125,63 |
| Dünya bitiş stoku | 73,19 | 73,19 | 73,19 |
| ABD üretimi | 18,59 | 18,59 | 18,59 |
| ABD tüketimi | 3,30 | 3,30 | 3,30 |
| ABD ihracatı | 15,00 | 15,00 | 15,00 |
| ABD bitiş stoku | 4,40 | 4,40 | 4,40 |
| Çin tüketimi | 41,50 | 41,50 | 41,50 |
| Hindistan üretimi | 27,50 | 27,50 | 27,50 |
| Brezilya üretimi | 11,00 | 11,00 | 11,00 |

[USDA düzeltme kaydı](https://www.usda.gov/historical-changes-revisions), 14 Aralık
repostunu süt tablosundaki değişiklikle açıklar. PDF basılı sayfa 33'te 2019
Aralık fat-basis ihracatı **11,0 → 10,0**, iç ticari kullanım **215,3 → 216,3**
billion pounds olarak doğrudan doğrulandı. Cotton basılı sayfa 27 ile iki süt
sayfası ve üniversite tablosu görsel olarak incelendi. Her iki PDF kapağı da
**11 Aralık** diyor: kapak tarihi dosya sürümünü veya fiilî teslimi belirlemiyor.

## Saat kanıtı neden kurulamadı?

[University of Tennessee kopyasının](https://news.utcrops.com/wp-content/uploads/2018/12/Monthly-Crop-Outlook-12-11-18.pdf)
basılı sayfa 6'sındaki on temel değer de eşleşiyor. Ancak `12-11-18` dosya adı
ve içerikteki rapor tarihi geçmiş erişim makbuzu değildir. PDF metadatası:

- CreationDate: `2018-12-12 08:19:42 -05:00` = **13:19:42 UTC**;
- ModDate: `2018-12-12 15:55:18 -06:00` = **21:55:18 UTC**.

İkisi de ilk karar kesiminden sonra. Metadata yayımlanma zamanı değildir; bu
bulgu, bir gün önce başka bir kopya bulunmadığını da kanıtlamaz.

Üç sınırlandırılmış CDX sorgusu `20181201–20181231`, status 200 filtresi ve
en fazla 50 kayıtla yapıldı: eski USDA PDF URL'si ve Cornell PDF URL'si
18 saniyede timeout; üniversite PDF URL'si HTTP 200 / boş indeks döndürdü.
URL/parametre/sorgu zamanı, hata veya ham yanıt saklandı. **Timeout boş indeks
sayılmadı; boş indeks de yayımlanmama kanıtı sayılmadı.** Arama burada durdu.
Önceki kaydedilmemiş sorgular bu teslimatın tekrarlanabilir kanıtına eklenmedi.

Güncel ESMIS sayfasındaki `12:00Z`, PDF oluşturma saati ve CSV `ReleaseTime`
erişim kanıtına dönüştürülmedi. Bir tarih etiketi, rapor takvimi, değer vintage'ı
ve ingestion zamanı farklı şeylerdir. Onay sözleşmesi değiştirilmedi:
`availability_verified=false`, `available_at=null`, kabul edilen tarihsel satır **0**.

## Karar ve yalnız sonraki adım

Bu tek olayda **Cotton revizyon kontaminasyonu ana şüpheli olmaktan çıktı**;
95 raporun bütün gerçek erişim saatleri doğrulanmış olmadı. Sonucu “WASDE'de
sinyal yok” veya “leakage tamamen çözüldü” diye okumak yanlıştır.

**Sonraki tek iş:** sayısal olarak doğrulanan bölgesel WASDE'nin sabit Cotton
çekirdeğine artımlı katkısını ölçen **bir** küçük CPU ablation'ının manifestini
önceden kilitlemek. Tarihsel saat kanıtı bulunamadığı açık kalacak; çalışma
erişilebilirlik varsayımına bağlı duyarlılık olarak ayrı tanımlanacak. Gerçek
erişimi doğrulayan mevcut kabul kapısı aşılmayacak, varsayımlı kaynak için sahte
onay makbuzu üretilmeyecek. Önce `python ml/history.py check --query wasde`
ile eski balance/text T+5 testleri karşılaştırılacak; onlar bu 26 alanlı adayın
T+1 testi değildir. Manifestte saat varsayımı, gecikme duyarlılığı, ortak
origin'ler, yalnız eğitimde preprocessing, olgunlaşma/purge ve sabit MAE/yön
ölçütleri kilitlenmeden fit başlamaz. 95 rapora arşiv saati araması veya büyük
model grid'i kendiliğinden açılmaz.

## Koruma ve yeniden kontrol

**2.190 eski dosya** hash'i yeniden doğrulandı, değişiklik **0**; ilk 95 raporlu
aday, eski sonuçlar ve sicil kayıtları korunur. Kaynak/test/runtime kodu
değişmedi. Çalışma ML kimliği:
`82a4d6fa5d06d8d6ecea6108da86ebac362d51235bbbd739cee63e9496197e21`.
Bu bir sıfır-fit kaynak incelemesidir; piyasa performansı bakımından **INCONCLUSIVE**.

Makine tarafından okunabilir [kanıt özeti](../research/evidence/wasde-clock-case-20261008.json)
ve [ek Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-clock-case-20261008)
iki orijinal PDF/XML/sayfa + ingestion makbuzlarını, üniversite PDF'sini,
sınırlı sorgu makbuzlarını, görsel kontrolleri, çalıştırılan doğrulama betiğini
ve ML snapshot'ını bağlar. Bunlar tarihsel saat için başarılı onay makbuzu değildir.
