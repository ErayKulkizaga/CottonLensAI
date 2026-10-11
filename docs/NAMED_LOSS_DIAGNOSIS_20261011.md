# Named T+5: öğrenilen kayıpta gizli beceri var mı?

**VERIFIED — mevcut ham tahminlerde yok.** PR42'nin iki sabit XGBoost kolu,
fiyat-MAE'nin yanında optimize edilen standardized log-MSE'de de Naive'ın
gerisinde. Bu sonuç bütün loss/model tariflerinin başarısız olacağı anlamına
gelmez; yalnız mevcut tahminlerin MSE başarısının MAE tarafından gizlendiği
açıklamasını desteklemez. Araştırma/pratik hedef henüz sağlanmadı.

## Değişmeden incelenen kanıt

`named-loss-diagnosis-v1`, sonuç-sonrası **sıfır-fit** teşhistir. Kaynak kontrolü
`research-named-history-source-control-t5-v1` tamamlanmıştır; tekrar eğitilmez.
ML source `466e4d3f756178561aca5e1706384d0be7398a7cf01757bef34677777c1ed155`.
44 native model; kol başına 189 iç ve 246 dış origin. Dışarıda aynı 242 uygun
ve dört bilinmeyen tarih korunur. Yeni tahmin/fit/seçim sıfır; shrinkage
değişmez. Sonuçlar daha önce görülmüş 2023'ündür; bağımsız holdout, ilk vintage
ve tarihsel yayın saati ispatı değildir. Canlı 126 cohort skoru hesaplanmaz.

Her refit'in saklanan hedef ortalaması ve ölçeği, yalnız olgun eğitim
etiketlerinden bağımsız yeniden hesaplandı. Native standardized çıktı da
yeniden çıkarıldı. Matematiksel standardized hata `(r-p)^2 / sigma_fit^2`;
ortalama iptal olur, değişen sigma dönemleri farklı ağırlıklandırır. Native
float32 etiket/çıktı ile bu matematiksel hesap ayrı kaydedilir.

## Sonuç

Kazanç, ilgili loss'ta Naive'a göre yüzde azalmadır; pozitif daha iyidir.
MSE yüzdesi fiyat-MAE için kilitli %5 ürün eşiği yerine kullanılamaz.

| Kol / dönem | Fiyat-MAE | Log-MSE | Standardized MSE |
|---|---:|---:|---:|
| Mask / iç 0 | −%13,194711 | −%17,118858 | −%16,982343 |
| Mask / iç 1 | −%13,869615 | −%19,003469 | −%17,853139 |
| Mask / iç 2 | −%6,371814 | −%8,271324 | −%8,661210 |
| Mask / dış | −%0,493740 | −%3,814775 | −%3,866798 |
| Numeric / iç 0 | −%13,559021 | −%18,045483 | −%17,940131 |
| Numeric / iç 1 | −%9,543329 | −%11,226640 | −%10,519782 |
| Numeric / iç 2 | −%5,756410 | −%10,576019 | −%10,866366 |
| Numeric / dış | −%3,005765 | −%6,032827 | −%6,359112 |

Standardized MSE dış kazancı için 10.000 tekrar/seed42, tam 246 ordinal
slot ve dört bilinmeyende sıfır bookkeeping ağırlığıyla eski paired
moving-block bootstrap kullanıldı. Blok20/60 aralıkları:

- Mask: [−%9,696625, +%2,173027] / [−%9,369388, +%1,131365].
- Numeric: [−%15,803083, +%4,400188] / [−%15,305910, +%4,713909].

Aralıklar sıfırı içerir: küçük pozitif popülasyon etkisi dışlanmaz; pozitif
OOS beceri desteklenmez. İç dönemler yalnız betimseldir; dönem seçilmedi.
Native ve matematiksel loss'un en büyük mutlak satır farkı 2,045541e−6;
kazançların işaretini değiştirmez. Ölçek aralığı 0,035274–0,044467.

**STRONGLY SUPPORTED:** aynı hedef/kapasite düzeltmeleri hatayı azaltmış olsa
da, mevcut bilgi-temsil-tarif öğrenilen hedefte kalıcı bir üstünlük üretmiyor.
Sıfıra shrinkage bunun geçmiş fiyat-MAE'deki sonucudur. “Sadece loss'u
değiştirince gizli başarı ortaya çıkacak” yaklaşımı bu kanıttan çıkmaz.
Kaynakta hiç sinyal olmadığı veya bütün modellerin noise floor'da olduğu
sonucu da çıkmaz: PR40'ta kapasite Ridge'e göre ölçülebilir katkı sağlamıştı.

## Doğrulama ve korunma

870 satır, 44 bağımsız native replay/scaler kontrolü, 862 uygun satırın
Decimal/scalar kaybı, dört bağımsız scalar bootstrap aralığı doğrulandı.
Yeniden hash'lenmiş yanlış scale, tarih, native çıktı ve loss dört ayrı
kontrolde reddedildi. Analitik bir örnek, MSE iyileşirken MAE'nin bozulmasını
kontrolün gerçekten ayırt edebildiğini gösterir; sentetik piyasa becerisi
değildir. Eski source/model/tahmin/proof/trials byte'ları korunur.

Teşhis R1'de scalar `exp(float32)` fiyat hesabı mevcut float64 raporuyla
eşleşmedi ve durdu; R2 genişletmesi eski MAE'yi tam doğruladı. Decimal fiyat
çıkarma ve yüzde normalizasyonunun hata sınırları R3/R4'te operand
büyüklüğünden türetildi. Önceki protokoller/sonuçlar korunur; tolerans
değişikliği eğitim, tahmin veya sonuç seçimi için kullanılmadı.

Kanıt: `research/evidence/named-loss-diagnosis-20261011.json`.
[Checksum bağlı analiz arşivi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/named-loss-diagnosis-20261011)
ayrı PR42 parent restore'una bağlıdır; modelleri ikinci kez fit etmez.

## Tek sonraki bilimsel soru

Mevcut sabit XGBoost düzeni, bu named cohort'un örnek sayısı ve eksiklik
deseninde **zayıf ama pratik büyüklükte bilinen bir sinyali** öğrenebilir mi?
Önce geçmiş kontrol sicili taranarak aynı tarif için eşdeğer testin bulunup
bulunmadığı doğrulanmalı. Eski 3.380 Ridge zayıf-sinyal fit'i ve dört güçlü
XGBoost sentetiği bu soruya otomatik cevap sayılmaz.

Tek aday, mevcut XGBoost tarifi/cohort/seçim sabit tutulmuş, gerçek piyasa
etiketleriyle başarı iddia etmeyen sınırlı **sentetik güç kontrolü**dür.
Bilinen öğretici etki, null eşlemesi, fit bütçesi ve beklenen oracle-MAE
avantajı ön kayıtla kilitlenmeden hiçbir fit başlamaz. Böylece düşük sinyal
çıkarımı ile bu sınırlı tarifin zayıf etkiyi öğrenememesi ayrılır. Yeni piyasa
modeli, loss araması, grid, lag araması veya otomatik kaynak ekleme yoktur.
