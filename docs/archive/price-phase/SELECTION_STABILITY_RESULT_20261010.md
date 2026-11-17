# İç doğrulama kararlılığı — sonuç

**VERIFIED:** 20 dondurulmuş senaryodaki160 karar,1.440 eski iç fit makbuzu ve
30.240 geçmiş doğrulama tahmini kullanılarak yeniden kuruldu. Yeni fit0;
eski tahmin, ağırlık ve veri değiştirilmedi. OOS outcome'lardan yeni ağırlık
seçilmedi. [Protokol](SELECTION_STABILITY_PROTOCOL_20261010.md) ve
[kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/selection-stability-20261010).

| Tanı (80 yıllık seçim/koşul) | Null | Enjekte sinyal |
|---|---:|---:|
|Üç leave-one-block-out seçiminin tamamı aynı|25/80|21/80|
|Blok20: kayıtlı ağırlığın yeniden seçilme sıklığı medyanı|%51,94|%56,19|
|Blok60: aynı sıklık|%96,93|%98,20|
|Kazanan–ikinci normalized validation MAE skor farkı medyanı|0,1791 yüzde puan|0,2735 yüzde puan|

**STRONGLY SUPPORTED:** bu üç kısa doğrulama bloğunda tek bir ağırlığın açık
üstünlüğü çoğu yıl sağlam değildir. Enjekte koşulunda en az bir doğrulama
bloğu çıkarılınca59/80 karar değişir. Bu, önceki zayıf kontrolün %49,56/%50
koruma sınırını kesin bir küçültme kusuru gibi yorumlamamayı destekler.
Değişken geçmiş seçimler, bütün piyasa başarısızlığını açıklamış değildir:
gerçek Texas ham tahmini de Naive'den kötüydü.

**VERIFIED:**63 günlük bir blokta60 uzunluğunda yalnız dört olası başlangıç
vardır. Dolayısıyla blok60'a bakıp %98 seçilme sıklığını "ağırlık güvenilir"
diye yorumlamak yanlış olur. Bu koşullu resampling tanısı, model yeniden
eğitiminin belirsizliğini veya piyasada başarı olasılığını ölçmez. Skor farkı
aralıkları kazanan seçimine göre düzeltilmiş formal test değildir.

Her160 seçimde kayıtlı mean-of-three-normalized-price-MAE ve küçük ağırlığa
eşitlik çözümü yeniden üretildi. Makbuz kimliği aynı olgun geçmiş eğitim
kohortundan kuruldu; train hedefleri chunk öncesinde, validation hedefleri
outer karar öncesinde olgunlaştı. Kullanılan5.960 girdinin hash'i yayımlanmış
orijinal envanterle eşleşti. Kodun ilk okuyucusu ledger'ın `repeat_reason=null`
kimlik alanını atladığı için sonuç üretmeden durdu; düzeltilen okuyucu ayrı
`analysis-r2` çıktısıyla çalıştı. Eski deneylere ilişkin hata tespiti değildir.

**Karar:** zayıf kontrolün RAW_ONLY_RECOVERS sınıfı değiştirilmez; Naive
korunur, yalnız bu tanıyla küçültme kaldırılmaz veya doğrulama penceresi
OOS sonucuna göre ayarlanmaz. Aynı kaynak/model grid'i büyütülmez.

Sonraki ayırıcı aday, sabit piyasa tahminlerinde log-getiri ortalaması ile
fiyat-MAE'nin koşullu medyanı arasındaki farkı incelemektir. Sicil taraması
aynı tam bias-calibration kapsamını bulmadı; bu yenilik kanıtı değildir.
Önce arşivlenmiş eğitim hedefleri/çıkarımlarıyla, sadece olgun geçmişte
fiyat-MAE'yi minimize eden tek çarpan kalibrasyonu için kapsam ve ön kayıt
hazırlanmalı. Bu tanı yeni piyasa eğitimi veya canlı model yayını başlatmaz.
Gerçek bilgi yokluğu ile model/loss etkisi hâlâ ayrılmış değildir.
