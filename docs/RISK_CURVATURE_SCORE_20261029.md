# Aynı risk bilgisindeki tek eğrilik yönü: sıfır-forecast-fit sonuç

**INCONCLUSIVE.** Mevcut log-lineer EWMA-relative HAR'ın kaçırdığı tek bir
karesel yön, aynı gerçek AMS hedefleri ve mevcut native tahminlerden sınandı.
Yeni tahmin modeli eğitilmedi; orijinal tahminler ve kayıplar değiştirilmedi.

| Ölçüm | Sonuç |
|---|---:|
| 2016–2023 ordinal / paired origin | 2.006 / 1.989 |
| Ortalama yerel iyileşme score'u | −0,02157805 |
| Blok20 %95 aralık | [−0,09117358; +0,04454094] |
| Blok60 %95 aralık | [−0,08773870; +0,04402773] |
| Pozitif / negatif yıllar | 3 / 5 |
| En büyük %1 EWMA kaybı çıkarılınca score | +0,00968581 |

İki aralık da sıfırı içerir; uç gözlemler çıkarılınca nokta işareti değişir.
Bu yönden yeni beşinci parametre/karesel HAR eğitimi veya daha büyük model
gerekçesi çıkmadı. Bütün doğrusal olmayan ilişkilerin yokluğu sonucu çıkarılmaz.
Score **QLIKE kazancı veya yüzde başarı değildir**; test edilmiş yeni tahmin yoktur.

## Tam olarak ne ölçüldü?

Her mevcut dış modelin geçmiş eğitiminde, üç relative-HAR özelliği kendi
native training mean/std değerleriyle normalize edildi. Relative-HAR5'in
karesi, aynı geçmişteki dört affine sütuna karşı yalnız feature'larla
ortogonalize edilip artık RMS'iyle ölçeklendi. Bu 97 label-free doğrusal
cebir dönüşümüdür; forecast model fit'i değildir. Test dönemi feature'ları
ve etiketleri dönüşüm katsayılarını değiştirmez.

Mevcut ham tahmin `p`, bu tek yöndeki olası perturbasyon
`p_gamma=p*exp(gamma*phi)` ise, ölçülen score:

`s = phi*(actual/p − 1) = −dQLIKE/dgamma | gamma=0`.

Her iki işaret analizden önce kabul edildi. Kural: iki 20/60 aralığı tamamen
aynı tarafta; en az6/8 yıl o yönde; top1% duyarlılığı da aynı yönde olmalı.
OOS gamma veya yeni feature seçilmedi. Kurallar önce yerel contract/source
kimliğiyle donduruldu; bu, haricen zaman damgalı bağımsız bir ön kayıt değildir.
Görülmüş araştırma yılları ve tarihsel UTC/ilk-vintage belirsizliği korunur.

Kaynak: `ce7e8ee2408d6f8e0972cd374569038b61878618a6d5b57d620089205cfc45d2`.
Tasarım: `e16fbecac11b910005f6cb952ad01a066ea2aeb9170a1f8abad0b27326cc60c2`.
Çalışma: `output/risk-curvature-score-20261029/execution-v2/`.
V1 yalnız hazırlıktır; açık sıfır-forecast-fit guard/lint düzeltmesiyle
analiz başlamadan superseded oldu; orada tahmin/score sonucu üretilmedi.

Bağımsız verifier97 mevcut dış modeli hash/native coefficient'tan doğrular,
feature projeksiyonlarını ayrı pseudoinverse hesabıyla kurar, 1.989 score'u
Decimal ile hesaplar ve iki paired-difference bootstrap aralığını tekrarlar.
En büyük basis farkı5,33e−15; score farkı3,55e−15. Dört yeni regresyon testi
ve Ruff geçti. Raporlar ve model payload'ları eski halleriyle korunur.

## Bir sonraki veri sorusunu daraltan bulgu

Risk modeli yalnız `har_1/har_5/har_22` kullanıyor; bu aynı pamuk geçmişinin
başka bir sıkıştırmasıdır. Dondurulmuş veri tablosunda DXY/WTI getirileri,
kaynak tarihleri ve varsayımsal availability saatleri zaten bulunuyor.
Yeni risk hedefinde bu kaynakların katkısı, eski fiyat/T+5 deneylerinden
çıkarılamaz. Kullanılmamaları bu kontrollü çalışmada kod hatası değildir;
kontrolün bilinçli dar bilgi kümesidir.

Sadece kaynak kurulabilirliği denetimi: 3.520 Cotton satırının bütün3.519
ardışık endpoint'i takvimde korunuyor. Mevcut, önceki Cotton gözlemine
hizalanmış DXY getirilerinden beş gözlemlik kare-getiri ortalaması2.006
originin tamamında kurulabiliyor. Kayıtlı varsayımsal availability saatleri
karar sonrasında değil; azami kaynak yaşı2gözlem. Bu sonuç DXY sinyali,
gerçek publication UTC veya model kabulü değildir.

Bu getirilerin ham kaynaktan yeniden kurulması [ek kaynak denetimiyle](DXY_RISK_SOURCE_AUDIT_20261029.md)
tamamlandı: 3.520 Cotton fiyatı/10.531 DXY return scalar'ı eşleşir; aynı
girdi2.787benzersiz train origin'inde de kurulabiliyor. Saat/vintage varsayımı
korunur; katkı henüz ölçülmedi. Tek sonraki iş sabit DXY-risk katkısı için
sıfır-forecast-fit QLIKE score tasarımıdır. Yeni dataset/model/grid açılmaz.
Mevcut pamuk bilgisinin dönüşümlerini tekrar tekrar büyütmek yerine,
gerçekten farklı bilginin erişip erişmediği sınanmalıdır.

[Private curvature sürümü](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/risk-curvature-score-20261029)
1.548 dosya/44.065.505 mantıksal baytı taze geri kurdu; exclusion0.
ZIP SHA-256:`1f202822d76aba4f6f4dfc45afe656a599e271769eed70f392db9bd5d861b2ad`.
Geri kurulmuş standalone bağımsız verifier, geri kurulmuş girdilerle aynı
makbuzu bayt eşit yeniden üretti. Eski207sicil aşaması/188kanıt dosyası ve
fit sicilinin bytes'ı korunur; bu iki yeni sıfır-forecast-fit kayıtla sicil
209aşama/551tarif/31.463fit kopyasıdır. Mevcut modeller yeniden eğitilmedi.
