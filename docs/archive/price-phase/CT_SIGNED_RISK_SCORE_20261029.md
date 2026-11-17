# CT işaret bilgisi: koşullu QLIKE tanısı

**INCONCLUSIVE — EWMA üstünlüğü gösterilmedi.** Bu kayıt yeni bir model
eğitmez. Önceki işaret incelemesindeki negatif ilişkiyi başarısız pozitif
hipotezle birlikte okuyup, mevcut modelin hatasında sınar. “Pozitif ilişki
bulunmadı” ile “işaret bilgisi yok” aynı sonuç değildir.

## Değişmeyen kanıt

Tamamlanmış `ct-ewma-har-full-year-t1-v1` çalışmasının 169 native fit'i,
kaynak snapshot'ı ve kayıtlı tahminleri doğrulandı; tekrar eğitim yapılmadı.
2016–2023, 2.006 T+1 origin, hedef/gerçekleşme tarihleri, kare getiri ve
EWMA λ=0,94 korunur. AMS annual tablosu bu analizde kullanılmaz.

97 dış işin her birinde yalnız geçmiş özelliklerden şu dönüşüm kurulur:

- İşaretli şok: `sign(r_t) * r_t² / EWMA_t`.
- Kontroller: intercept, üç relative-HAR ve `r_t² / EWMA_t`.
- İşaretli şokun bu kontrollere geçmiş projeksiyonundan kalan kısmı, geçmiş
  residual RMS ile ölçeklenir; hiçbir hedef projeksiyona girmez.
- İncelenen score: `phi * (actual_variance / native_raw_variance - 1)`.

Bu score, `native_raw_variance * exp(gamma * phi)` yönündeki QLIKE kaybının
gamma=0 noktasındaki negatif türevidir. Gamma öğrenilmedi; bu sayı QLIKE
kazancı veya kullanılabilir bir tahmin değildir. İki işaret de hesap öncesi
kabul edildi. Önceki negatif ilişki görülerek seçilen bir geliştirme analizi
olduğu, bağımsız holdout olmadığı açıkça kaydedildi.

## Sonuç

| Ölçüm | Değer |
|---|---:|
| Ortalama score | −0,24122132 |
| Blok 20, %95 aralık | [−0,50729372; +0,19320329] |
| Blok 60, %95 aralık | [−0,54228187; +0,14232997] |
| Negatif / pozitif yıl | 5 / 3 |
| En büyük %1 EWMA kaybı çıkarıldığında | −0,01573317 |

Yıl sınırlarını koruyan 10.000 tekrar/seed 42 kullanıldı. Her iki aralık
sıfırı içerir; 6/8 aynı işaret koşulu da sağlanmadı. Sonuç, eski ex-post
negatif covariance'ı sağlam tahmin avantajına dönüştürmez. İşaret bilgisinin
her temsil/modelde yararsız olduğunu veya üstünlüğün imkânsızlığını kanıtlamaz.

## Uç günün gerçek kaynak denetimi

Tanımlayıcı takip, en büyük EWMA kaybını taşıyan origin'i seçti; ana kohortu
veya sonucu değiştirmedi. **23 Haziran → 24 Haziran 2022**, eski ham CT-HAR
çalışmasının toplam net QLIKE kazancının **%91,5524**'ünü taşır:

- Bütün 2.006 günün ham net kazanç toplamı: +35,05411350.
- Tek günün katkısı: +32,09288469.
- O gün çıkarıldığında ham ortalama katkı: +0,00147692. Bu, yeni ana sonuç
  veya başarı metriği değildir; eski ana sonuç korunur.
- Aynı gün, yeni işaret score'unun net toplamının %85,0649'unu taşır.

İki eldeki tarihli MP_CN001 raporunun literal tablosu kontrol edildi:

| Rapor kontratı | 23 Haziran fiyatı | 24 Haziran fiyatı | Log getiri |
|---|---:|---:|---:|
| Jul-22 | 136,32 | 103,76 | −0,27292452 |
| Oct-22 | 109,01 | 107,15 | −0,01720990 |

Jul-22 fiyatları eldeki CT close'larıyla yuvarlama toleransında eşleşir.
CT hacmi 5.807'den 174'e düşer. Bu gözlemler yakın vadeye özgü fiyat/hacim
olayını gösterir; olayın ekonomik nedenini ispatlamaz. İki rapor da **aynı
Jul-22** etiketini taşırken otomatik olarak “continuous roll yapay sıçraması”
demek desteklenmez. Aynı yayıncıdaki iki tarih, bağımsız ICE veri doğrulaması
veya ilk-vintage kanıtı değildir; ortak kaynak hatası da dışlanmış değildir.
İkinci kontrat getirisi ana hedef yerine geçirilmedi.

## Doğrulama ve koruma

Bağımsız denetim close'tan 3.520 satırın getiri/HAR/EWMA'sını Decimal ile
yeniden kurdu; ayrı pseudoinverse ile 97 geçmiş projeksiyonunu ve 2.006
QLIKE türevini doğruladı. İki CI ayrı sum-difference bootstrap ile eşleşti.
Uç-gün eki ayrıca 2.006 hedefi ve 4.012 QLIKE loss scalar'ını yeniden kurdu.
16 ilgili test ve Ruff geçti. UI/API değiştirilmedi; UI build açılmadı.

Hesap sırasında `Ledger.run`, learner fit girişleri ve `scipy.optimize.minimize`
çağrıları hata verecek şekilde kapatıldı. Eski native dosyalar, fiyatlar,
tahminler, fit sicili ve eski kanıt hash'leri korunur. V1 kimlik kapsamı ve
V2 tuple adaptör hatası, hiçbir score/fit üretmeden ayrı kayıtlarla kapandı;
nihai analiz `execution-v3`'tür.

Private [kanıt sürümü](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-signed-risk-score-20261029)
1.800 ana + 10 ek dosya olarak taze geri kuruldu. Ana result/CSV/dönüşümler,
bağımsız makbuz ve beş uç-gün ek dosyası sıfır-fit/offline yeniden hesaplamada
byte eşit çıktı. Ana SHA `face179c62f22447c1e02166e8a193da27c61f835cbb3425708d21d72c76263d`,
ek SHA `fb76db369a31c09e0ec811328b33ed1aab0253310aff2f0bf5975d7e953352b3`.

## Tek sonraki bilimsel ayrım

Henüz ölçülmeyen soru, işaret katsayısının yalnız geçmişte öğrenildiğinde
gerçek tahmin katkısı sağlayıp sağlamadığıdır. Tek sabit karşılaştırma,
aynı üç HAR'a birinde unsigned şok, diğerinde signed şok ekleyen iki
**eşit beş parametreli** kol olabilir; işaret sonradan elle çevrilmez.
Mevcut dört parametreli 169 kontrol fit'i tekrar edilmez. Aynı cohort,
target, EWMA, olgunlaşma, selection/refit ve üstünlük kapıları korunmalıdır.

Bu tarif henüz hazırlanmış/kilitlenmiş veya çalışmış değildir; önce ayrı
kimlik, geçmiş kontrolü, giriş/native öğrenme doğrulaması ve ön kayıt gerekir.
OOS score anlamlılığı, bu ayrı tahmin testinin başarı sonucu yerine geçmez
ve otomatik fit açmaz. Yeni grid/ufuk veya GPU programı yoktur. Named kaynak
onarımı ayrıca kabul edilmeden named-market eğitimi başlamaz. Saatlik görev
kapalı kalır; ölçülmüş piyasa başarısı hâlâ yoktur.
