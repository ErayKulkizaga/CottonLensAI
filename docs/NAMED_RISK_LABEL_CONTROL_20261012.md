# T+1 risk: aynı modelde yalnız eğitim etiketi kontrolü

## Karar

**VERIFIED — hedef uyumsuzluğu bu deneyde modelin hatasını artırıyor.**
Sabit Log-HAR, CT etiketleri yerine origin'de seçilen ikinci kontratın
kare getirisiyle eğitilince QLIKE 0,07117631 iyileşti; CI20/60 pozitif.
Bu, bir yıllık ve erişim/vintage varsayımına bağlı **etiket mekanizması**
bulgusudur. **EWMA'yı geçen risk tahmini başarısı değildir.**

Named-trained HAR hâlâ doğru named EWMA'dan ortalamada 0,03363883 daha
kötü. Geçmiş doğrulama iki model kolunda da ağırlık0 seçti; seçilmiş
tahmin referansla aynıdır. Bunun [0,0] aralığı sinyal yokluğunu kanıtlamaz.
Colab/GPU kapasite artışını haklı çıkaran model üstünlüğü henüz yok.

## Kilitli tasarım ve kanıt

- Profil `named-risk-label-control-t1-v1`; kaynak
  `42221c6e9b71a39588f2299fddd1325a1c047bb933387deda7aea565d79df203`.
- 2023, 246 parent origin; eksiklik gerekçeleri korunur; 242 uygun paired
  origin. Eğitimde en az576 örnek; aynı expanding train, 3×63 geçmiş
  validation, 21 Cotton gözleminde refit, beş-gözlemlik olgunlaşma.
- İki kolda da aynı üç CT HAR özelliği ve aynı tarihler. Yalnız train'deki
  `variance_target_1` değişir. İki kol aynı named T+1 gerçekleşene karşı
  skorlanır; ham ve geçmişte seçilmiş varyans karışımı ayrı tutulur.
- Referans lambda0,94: origin'de seçilen kontratın **kendi** geçmişinden;
  tam ardışık Cotton fiyat çiftleri; en az22 gözlenen getiri. Eksik çift
  getiriye dönüştürülmez; gün aşırı fiyat farkı günlük getiri sayılmaz.
  Gecikmiş çift yalnız varsayımsal erişim saatinden sonra kullanılabilir.
  Eksik çiftte state korunur; lambda gözlenen getiri başına uygulanır.
- 44 piyasa +1 sentetik fit tamamlandı; sentetik kontrol maksimum bağıl
  hatası3,33e−15. Sentetik sonuç piyasa becerisi değildir.
- 2016–2023 içinden yalnız2023 protokolü destekledi; 2024+ kullanılmadı.
  Sekiz-yıl6/8 kapısı ve bağımsız holdout iddiası yoktur. Tarihsel UTC
  ve ilk vintage doğrulanmış değildir; aday kaynak bayrakları değiştirilmedi.

## Sonuçlar

Delta `QLIKE(referans)-QLIKE(aday)`; yüzde değil. Label mekanizmasında
referans CT-trained HAR; diğerlerinde aynı-kontrat EWMA.

| Karşılaştırma | Ortalama delta | CI20 | CI60 | En büyük %1 EWMA kaybı çıkarılmış delta |
|---|---:|---|---|---:|
| CT-trained → named-trained HAR | +0,07117631 | [0,02492001;0,12265010] | [0,02273516;0,11680117] | +0,09447220 |
| Named ham HAR → EWMA | −0,03363883 | [−0,07668998;0,01175033] | [−0,06755747;−0,00077030] | −0,05827275 |
| CT ham HAR → EWMA | −0,10481514 | [−0,18344977;−0,02408393] | [−0,17523324;−0,02820074] | −0,15274495 |
| Named seçilmiş → EWMA | 0 | [0;0] | [0;0] | 0 |

10.000 tekrar/seed42, blok20/60; 246 tarih slotu korunur, uygunsuz
origin'in weight0 olması zamanı sıkıştırmaz. Trim ana hükmü değiştirmez.
İyileşme yalnız ham model hatasını azaltır; başarı için referansı geçmek gerekir.

## Doğrulama ve paketleme düzeltmesi

30 ilgili test ve Ruff geçti. Native44 model/fit kimliği, kaynak payload
hash'i, tahmin üretimi ve geçmiş weight seçimi raporda yeniden doğrulandı.
Bağımsız Decimal kontrolü242 actual/EWMA,1.210 scalar kayıp, sekiz CI,
22 cutoff'ta17.248 eğitim satırının olgunluğunu ve sekiz girdi hash'ini doğruladı.
Ek13 full-year regresyon testiyle toplam43 ilgili ML testi geçti.

[Sonuç/teslimat makbuzu](../research/evidence/named-risk-label-result-20261012.json)
ve [private Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/named-risk-label-result-20261012)
573 dosyayı taze indirme/native restore ile doğruladı. Sonuç, paired CSV
ve bağımsız makbuz byte-identical yeniden üretildi; replay yeni fit yapmadı.

İlk ön kayıt paketine yanlışlıkla alınan tek generated `.pyc`, import
sırasında değişti. Kontrol eğitimi durdurdu; source/veri değişmedi. İlk
paket korunur; R2 yalnız bu geçici önbelleği dışarıda bırakır. R2'nin291
dosyası taze GitHub indirmesi/native restore ve `python -B` doğrulaması
geçmeden hiçbir fit başlamadı. [Ön kayıt kanıtı](../research/evidence/named-risk-label-preregistration-20261012.json).

## Tek sonraki bilimsel soru

**Temiz etiket yeterli değilse, HAR'ın CT geçmişiyle beslenmesi kalan
uyumsuzluk mu?** Referans şu an aynı kontratın geçmişini biliyor; modelin
özellikleri hâlâ sürekli CT geçmişidir. Bu, kapasite eksikliğiyle aynı şey
değildir. Önce mevcut aynı-kontrat fiyatlarıyla üç HAR özelliğinin
kurulabilirliğini denetle; sonra yalnız bu özellikleri değiştiren sabit
HAR kontrolünü, aynı named eğitim etiketi ve güçlü EWMA ile ön kaydet.
Eski T+5 price-history çalışması bu T+1 risk sorusunu sınamış sayılmaz.
Yeni grid, kaynak, ufuk veya GPU modeli gerekçelendirilmiş değildir.

Bu belge devam hedefini tamamlanmış ilan etmez; güçlü referansı geçen
tahmin ve Colab'a geçiş gerekçesi henüz kanıtlanmadı.
