# Sekiz yıllık CT risk kontrolü için zaman/veri kurulabilirliği

Sabit dört parametreli QLIKE HAR'ın origin EWMA0,94'e relatif öğrenilmesi
aynı 2016–2023 CT proxy risk kapsamına kurulabilir. Bu yalnız temporal/data
readiness'tır: model fit0, candidate prediction0, training_allowed=false.

Kaynak orijinal risk-audit history SHA
`2132162a410b4e12655d36a2ac3e9efd62a3fc0e49fdb08cb038d4cc328836d7`;
recorded risk forecasts'ları ve tüm2.006ordinal origin değişmez. Deterministik
EWMA yeniden hesaplanarak eski referansla eşleştirildi. Modelin üç alanı
`har_k-log(origin_ct_ewma)`, training hedefi `variance_target_1/origin_ct_ewma`
olabilir; EWMA çarpanı çıkarımda origin'de gözlenen state'tir.

Geçmiş189olgun validation/3×63,119warmup,`target_date_5<cut` koruması,
21Cotton-observation refit ve min500training korunur. Sekiz yıl toplam169iş
ister (2020:22; diğer yıllar21). En küçük train1.191; yıllar arasında
1.191/1.441/1.692/1.943/2.195/2.448/2.700/2.951. Her cutoff'un train'ı
ve origin bucket sınırı bağımsız olarak denetlenir.

`research/ct_ewma_har_readiness.py` mevcut planner'ın eligibility arayüzüne
yalnız geçici alias sağlar; bu alias'lar dosyaya veya learner'a yazılmaz.
Orijinal tüm sütunlar byte/float değerleriyle korunur. CT hedefi named
settlement hedefi diye yeniden adlandırılmaz. Inline V1 taslak ve source-bound
V2 ayrı korunur; V2'nin çalıştırılan kaynak byte'ları hash ile saklanır.

Bu soru tamamlanan named2023/22fit kontrolüyle aynı scope/recipe değildir.
Sicilde QLIKE/HAR/T+1 ilgili dört named fit tarifi ve risk denetimi incelendi;
eşleşme yokluğu tek başına yenilik ispatı değildir. Yeni kontrol bir candidate
ile EWMA'yı karşılaştırır; farklı popülasyon ve baseline state nedeniyle
“saf loss katkısı” iddiası yapılamaz. Asimetri işareti veya CFTC bu adayın
girdilerine eklenmez; başarısız screen'ler başarı diye yeniden sınıflanmaz.

Bir pilot ön kaydı yapılırsa model native receipts/Ledger, frozen source/data/
recipe, bounded CPU budget ve taze GitHub restore gerektirir. GPU/sequence
Colab-only; kaynak discovery veya readiness Colab geçişi değildir. Risk
karşılaştırması için iki20/60candidate-minusEWMA QLIKE CIüstü<0,
en az6/8negatif yıl delta ve top1%baseline-loss trimdelta<0 gerekir.
Fiyat5%MAE ve53/55direction kapıları değişmez; risk başarısı onları geçirmez.

İlk-vintage/market decision UTC varsayımları ve CT sürekli fiyat proxy'si
sınırlaması korunur. 2024+ seçime giremez; bağımsız holdout iddiası yoktur.
Sonraki iş aynı sabit aday için runnable/native-verifiable yürütme ve ön kayıt;
mevcut readiness dosyası eğitim yetkisi değildir.

Yerel kanıt: `output/ct-ewma-har-readiness-20261013/v2-code-bound/`.
