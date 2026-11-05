# Naive-MAE audit ve deney ön kaydı — 5 Kasım 2026

## Karar ve mevcut açıklama

Yeni piyasa eğitimi başlamadan önce hedef ve tanı kontrolleri geçti. Eski
başarısızlık tek bir veri/karşılaştırma hatasına indirgenemiyor. Kaydedilmiş
ham modeller hatayı büyütmüş, geçmiş küçültme zararı azaltmış; kaynak/temsil
katkısı geçmişten sonraki yıla kararlı taşınmamış. Loss düzeltmesi kontrole
katkı sağlamış fakat Naive'ı geçirmemiş. Bu, her modelde/sahada sinyal yokluğu
kanıtı değildir. En savunulabilir sonraki test düşük boyutlu koşullu medyandır;
bu testin başarı olasılığı hakkında ölçülmüş bir iddia yoktur.

## Dosyadaki taleplerle mevcut kanıtların eşlemesi

| İstenen kontrol | Yeniden doğrulama / sonuç / sınır |
| --- | --- |
| Repo/geçmiş envanteri | Sicil234aşama/563tarif/32592fit kopyası geçerli.665yerel payload hash'i eşleşti; yardımcı open/close proxy'nin6referansı sağlanan yerel köklerde yok. Kaybolmuş uzak kanıt diye adlandırılmadı. |
| Önceki modeller/feature'lar | Sicil ve trials tarifleri korunuyor. Return-path,Ridge/ElasticNet/ağaç,statistiksel,USDA/WASDE/NASS,zamanlama ve loss kontrolleri incelendi; aynı grid açılmıyor. |
| Fiyat alanı/roll | Eski fiyat-semantiği kaynak mutabakatı korunuyor. Close high–low dışında olabilir; kırpma yok. Dated günlük tanıklar ile annual AMS kimlik hataları ayrı. Named full-year kaynak kusurları bütün CT fiyatlarını yanlış kılmaz. |
| Hedefler/takvim | Hash-bağlı4205rawCotton barından3520tarihçenin iki hedefi/tarihleri yeniden kuruldu. Canonical split9d964b2c aynı2006origin/ufuk. Gerçek borsa seansı/vintage garantisi değil. |
| Leakage/scaling/seçim | Eski676Ridge native state train-only yeniden kuruldu; H5etiketi her fit öncesinde olgun.32yıllık çıktı/8024CSVsatırı kayıtlı native tahminlerle eşleşti. Yeni sentetik/prefix/maturity testleri mevcut. |
| Legacy LSTM/TCN | Ortak-origin LSTM düzeltmesi önceki artı sonucu geri alıyor; tam eski LSTM tahmin dosyası olmadan pairedCI yok. Eski TCN learning-control passed etiketi tek başına yeterli değil. Yeni iki aile gerçek sentetik kontrollere tabi. |
| DXY/WTI/WASDE | Mevcut zamanlama ve bölgesel deney sonuçları korunur; yeni kaynak/grid yok. Bar tarihi gerçek yayın/ilk vintage kanıtı değildir. |
| Naive diagnostikleri | Yıl/ay/volatilite/trend,tail,ACF ve feature-association tabloları/grafikler üretildi. Kesimler yalnız refit öncesi mature geçmişten; realized büyük hareket ve olay grupları açıklayıcı. |
| WASDE/roll tanıları |95bilinen rapor tarihi; diğer günler unknown. Roll için tüm h-adımlık kaynak yolu doğrulanmalı; boş tanıklar bilinmiyor kalır. T+1:961same/20transition/1025unknown;T+5:847same/94transition/1065unknown. |
| Hipotez/model | Tek yeni tarif:5fiyat özelliği,Quantile(.5,alpha=.01) ve aynı-input Ridge(1),price_delta. Önceki24featureMAE-ağaç/globalkalibrasyon ile aynı değil; yalnız loss etkisi iddiası yok. |
| Walk-forward/ablation | Mevcut119warmup,5purge/H5maturity,3x63iç,21refit;338market/4synthetic sınırı. Ablation yalnız birincil CONTINUE halinde. |
| Bağımsız doğrulama/güvenlik |2016–2023/seen2024+holdout değil. Eski artifact'ler değiştirilmedi; inference/görevler kapalı/korunuyor. Hedef veya gate değiştirme yok. |

## Naive bulguları

T+1MAE1.020952,medyan mutlak hata0.660004cent/lb. T+5MAE2.357184,
medyan1.599998.2022T+1MAE2.181315;2016MAE0.655320. Signed T+1lag1ACF
0.074513,absolute errorACF0.260735. Volatilite kümelenmesi yön becerisi değildir.
Signed/absolute ACF1/5/20 ve geçmiş feature Spearman ilişkileri CSV/JSON'dadır;
çoklu/sonradan tanılar bağımsız anlamlı sinyal elemesi gibi kullanılmadı.

## Dondurulmuş ilk deney

[Onaylanan plan](NAIVE_MAE_PLAN_20261105.md) aynen korunur. Birincil seçilmiş
Quantile,ikincil aynı-input Ridge. 5price feature'a ortak preprocessor yalnız
train-only medyan/std ve eksiklik göstergeleri ekler; bu dış bilgi değildir.
Birincil geçiş >=5%MAE,>=53%tüm-origin yön,>=6/8yıl ve blok20/60CIalt>0.
4sentetik kontrol ayrı tutulur. Başarısız/belirsiz geçerli tarif STOP;
veri/metod hatası REVISE. Bu kapılar bağımsız ileri başarı değildir.

Kaynak ve tasarım kimlikleri aşağıdaki ön kayıtta; ham veri ve source snapshot
ilk fit öncesi kopyalandı. Yeni modele ilişkin sonuç henüz bu belgeye dahil değildir.
[Ön kayıt](../research/evidence/price-only-median-prereg-20261105.json).
