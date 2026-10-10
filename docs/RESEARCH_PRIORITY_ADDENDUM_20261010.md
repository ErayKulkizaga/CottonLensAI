# 10 Ekim araştırma karar eki

Bu ek [mevcut planı](RESEARCH_DATA_REENTRY_PLAN_20261005.md) değiştiren yeni bir
araştırma programı değildir. Tarihsel adımlar, eşikler, sicil ve kaynak kabul
kuralları korunur. Güncel sıra için [STATUS](STATUS.md), sınırlar için
[AGENTS.md](../AGENTS.md) ve [araştırma sözleşmesi](TRADING_RESEARCH_CONTRACT_20261005.md)
esas alınır. Yeni model, grid, veri edinimi veya yayın izni vermez.

İncelenen kullanıcı belgesi: `CottonLensAI_AI_Arastirma_Yonlendirmesi_2026-10-10.md`.
SHA256: `a9ce7b447e6399af0aafb01aec8c1369a9ed1af837687b13e6ee100111878b97`.
Belge dış inceleme/öneridir; içindeki durum ve teşhis ifadeleri deney kanıtı değildir.
Bu incelemenin başlangıç commit'i `1f51b7ad521fc6bd35534cf695e9c06fdf304962`;
ML kaynak kimliği `d3ca00e498639dc266f34dafb60f2f24dbba5e66ee0deeda0d9c4c0e5b7f7313`.
İleri kayıt görevi farklı, sabit `44f3d16f…` kaynağını kullanır.

## GitHub ve belge arasındaki fark

**VERIFIED:** GitHub `main` hâlâ PR #9 merge commit'i
`c38e15191fbca22e0fd44f5c4488b6c1d50bf61a`. #10–#28 açık, birbirinin dalını
base alan zincirdir; #10'un base'i main. Açık PR, tamamlanmamış deney demek
değildir; sonuçlar main'e geçmiş de sayılmaz. Bu incelemede merge yapılmadı.

| PR / incelenen head | Gerçek durum ve karar |
|---|---|
| [#25](https://github.com/ErayKulkizaga/CottonLensAI/pull/25), `d220fb7bf155267caf73e5a0a5fc6536277eab26` | **Sonuçlandırıldı:** 169 skaler güncelleme, 0 yeni model fit'i, 2.006 T+1 origin; global kalibrasyon sabit tarifi kurtarmadı. Tekrar çalıştırılmayacak. |
| [#26](https://github.com/ErayKulkizaga/CottonLensAI/pull/26), `508c428568231d9b9ecbb6cc51d54030cf8d30f5` | **Sonuçlandırıldı:** 6.648 eski makbuz; MAE zaten denenmiş, eski arama loss etkisini tek başına ayırmamış. Yeni MAE grid'i gerekçesi değil. |
| [#27](https://github.com/ErayKulkizaga/CottonLensAI/pull/27), `8b69ef8e33f159599770ffdd310efa8a555bc810` | GitHub'da sıfır-fit ön kayıt; **yerelde 48 piyasa + 6 sentetik fit tamamlandı**. Bu incelemede 54 kayıtlı modelin çıkarımı doğrulandı; bağımsız bootstrap ve taşınabilir sonuç teslimatı henüz kapanmadı. |
| [#28](https://github.com/ErayKulkizaga/CottonLensAI/pull/28), `1f51b7ad521fc6bd35534cf695e9c06fdf304962` | Salt okunur gece kanıtı: yayın 0, missing 6. Uyku olayı pencereyi kapsıyor; sonraki görev isteği reddedilmiş. Bu piyasa performansı değil. |

#25–#27 head'lerinin kontrol rollup'ları 12 tamamlanan kontrol, #28'in rollup'ı
8 tamamlanan kontrol içeriyordu; başarısız kontrol görülmedi. #22'de bir kontrol
hâlâ devam ediyordu. Bu tarihli durum, yeni doküman PR'ının CI sonucu değildir.
Kod commit'i, ön kayıt kimliği, sonuç teslimatı commit'i ve çalışan snapshot
birbirinin yerine kullanılmaz.

## Teknik bulguların doğrulanması

| Belgedeki iddia | Kanıt ve düzeltme |
|---|---|
| PR #25 yalnız ön kayıt, sonraki iş 169 kalibrasyon | **Güncelliğini yitirmiş.** [Sonuç](PRICE_MAE_CALIBRATION_RESULT_20261010.md) ve [kayıt](../research/evidence/price-mae-calibration-20261010.json) tamamlanmış. Release ZIP digest'i GitHub'da `920d5586…e677` ile kayıtlı SHA'nın aynısı. Kalibre seçilmiş kazanç −%0,026081; kontrole katkı %0,001362; %5 hedefi kurtulmadı. 169 skaler güncelleme öğrenme/postprocessing'dir, 0 model fit'i demek 0 hesaplama demek değildir. |
| Loss–hedef uyuşmazlığı henüz hiçbir biçimde test edilmedi | **Bu kapsamda yanlış.** [#26](LEGACY_LOSS_LINEAGE_RESULT_20261010.md) altı MAE seçimi, üç price_delta+MAE seçimi buldu. Train-only affine standardizasyon, absolute loss'un medyan hedefini ortalamaya dönüştürmez. Eski karma arama kontrollü loss farkını ölçmüyordu; #27 bunu ayırıyor. |
| Texas/NASS, WASDE ve kaynak zamanı pratik hedefi kurtarmadı | **VERIFIED, sabit tariflere koşullu.** NASS D0 −%0,027444 /2.006 origin; WASDE D0/D1 −%0,522940/−%0,010733 /1.254 origin; saat müdahalesi −%0,153805 /2.006 origin. Bunlar farklı kohortları sıralamak veya bütün kaynaklarda sinyal yok demek değildir. [NASS](../research/evidence/nass-regional-result-20261009.json), [WASDE](../research/evidence/wasde-regional-result-20261008.json), [zamanlama](AVAILABILITY_CLOCK_RESULT_20261005.md). |
| Sentetik öğrenme ve seçim kararlılığı piyasa becerisi değildir | **VERIFIED.** Oracle/ham/seçilmiş ortalama kazanç %3,6054/%1,9551/%1,8752; enjekte koşulunda leave-one-block-out 59/80 karar değiştirir. Oracle etkisi bile %5'in altında; bu deney pratik %5 kapısının geri kazanım kanıtı değildir. Küçültmeyi kaldırma sonucu çıkarılmaz. [Zayıf sinyal](../research/evidence/weak-signal-result-20261009.json), [seçim](../research/evidence/selection-stability-20261010.json). |
| Bütün Close bozuk veya yalnız roll düzeltmesi çözüm olabilir | **NOT SUPPORTED.** 997 kabul edilmiş tablonun 996 Close eşleşmesi ve sınırları [fiyat/seans kanıtında](../research/evidence/price-semantics-20261005.json). Bilinen roll kapsamı bütün yılların kontrat kimliği veya tüm label'ların onayı değildir. |
| MAPE yeni metrik altyapısı gerektiriyor | **Gerekmez.** Mevcut `evaluation.evaluate` zaten `100 × mean(abs(actual_price−predicted_price)/actual_price)` hesaplıyor. Fiyat-MAE kazancı MAPE değildir. Eksik raporlama varsa yalnız mevcut tahminlerden eklenir; sıfır/negatif gerçek fiyat kabul edilmez, sessiz origin atılmaz. Eski karar ve kapılar yeniden sınıflandırılmaz. |
| Yeni gece gerçekten yayınlandı mı? | **Hayır.** 10 Ekim 11:10 Türkiye saati salt okunur durum: published 0 /missing 6 /required 126 /kalan 120. [Gece kanıtı](FORWARD_WINDOW_AUDIT_20261010.md). Naive/EWMA yayını olsa bile öğrenilmiş aday becerisi değildir. |

## Son kayıtlı deneyde fiilen doğrulananlar

PR #27'nin `research-paired-price-loss-control-v1` yerel kayıtları incelendi.
Ön kayıt/veri/kod kimliği, payload hash'leri, 48 piyasa ve ayrı 6 sentetik
modelin native XGBoost çıkarımı doğrulandı. İmputer/scaler ve target mean/scale
yalnız kayıtlı training satırlarından yeniden hesaplandı; piyasa training
etiketlerinde H1/H5 hedef tarihleri ilk test origin'inden önce. Üç seed ortalaması,
1.008 ortak origin/hedef, 2.016 ensemble /6.048 seed satırı ve bağımsız doğrudan
fiyat-MAE/yön hesabı eşleşti. 830 deney dosyasının önce/sonra hash'i aynı;
yeni fit 0. Bu, fit'in baştan yeniden üretimi veya tarihsel PIT kabulü değildir.

| Aynı T+1 kohortu | Naive fiyat-MAE kazancı | Yön | Kazanılan dönem |
|---|---:|---:|---:|
| MSE | −%1,975104 | %46,9246 | 1/8 |
| MAE | −%0,488298 | %48,4127 | 3/8 |

Birincil soru loss kollarının katkısıdır. MAE'nin daha az zarar vermesi,
Naive üstünlüğü anlamına gelmez. Sekiz blok **126-origin dönemidir**, tam yıl
değildir; 2.006-origin full-year kohortuna karıştırılmaz. Eski parametre/ağaç
sayıları sabit; bazıları çok küçük ve karma loss/hedef altında seçilmişti.
Genel model kapasitesi veya bütün MAE modelleri hakkında hüküm verilemez.
Ham kontrol/intervention çıktılarında shrinkage yok; buradaki başarısızlık
küçültmenin sinyali sıfırlamasına bağlanamaz. Altı paired aralığın bağımsız
replay'i ve kamuya indirilebilir sonuç arşivi henüz bu ekle tamamlanmış sayılmaz.

## En fazla üç engel; kök neden sınırı

1. **STRONGLY SUPPORTED — mevcut tahmin düzeltmeleri kalıcı OOS katkı vermiyor.**
   Ham Ridge zararlı; global bias düzeltmesi kurtarmadı; doğru fiyat-MAE loss'u
   olan kayıtlı tahminler de Naive'den kötü. Yalnız yanlış loss veya yalnız
   küçültme açıklaması yeterli değil. **En büyük bilinmeyen:** kullanılabilir
   bilgide zayıf sinyal ile sabit temsil/modelin onu çıkaramaması ayrışmadı.
   “Cotton tahmin edilemez” veya “daha büyük model çözer” kanıtlanmadı.
2. **VERIFIED — incelenmiş geçmiş bağımsız başarı kanıtı değil.** Aynı yılların
   tekrar incelenmesi, iç seçim kararsızlığı ve karma eski seçimler dış
   genellenebilirliği sınırlar. Bu sınır gerçek negatif MAE'yi kendiliğinden
   açıklamaz; yeni bir tarihsel kazanımı da bağımsız keşif yapmaz.
3. **VERIFIED — gerçek ileri yayın/erişim kanıtı eksik.** Görev kaydı boş;
   FAS specific-version ve diğer vintage/saat sınırlamaları sürüyor. Bu,
   gerçek-zaman ispatını engelliyor; geçmişteki bütün negatif sonuçların
   nedeninin timestamp olduğu gösterilmedi. Operasyon ve bilim ayrı tutulur.

## Sonraki tek bilimsel adım ve durma kapısı

**PR #27'nin mevcut çıktısını sıfır yeni fit ile sonuçlandırmak.** Kalan işler:
iki Naive karşılaştırması ve MAE–MSE karşılaştırmasının blok20/60, 10.000
tekrar/seed42 altı aralığını bağımsız hesaplamak; dönem/uç-hata metriklerini
doğrulamak; frozen kaynakla geçici dizinde rapor/CSV replay'i; ayrı sonuç
arşivini GitHub'dan geri indirip hash/çıkarım/replay doğrulamak. 48+6 tamamlanmış
fit yeniden çalıştırılmaz; 169 eski kalibrasyon da tekrarlanmaz.

Kaynak, olgunlaşma, origin/hedef veya replay farkı varsa bilimsel sınıf
kesinleştirilmez; yalnız doğrulama hatası incelenir. Kayıtlı karar kuralları ve
eşikler değişmez. Olumlu MAE–MSE katkısı, başarısız Naive kapılarını geçirmiş
sayılmaz. Negatif/belirsiz sonuç yeni grid'e dönüştürülmez. Sonuç teslimatı ve
deney siciline ek kayıt ayrı bilimsel PR kapsamıdır; bu plan PR'ı sicil/trials,
eski kanıt veya dondurulmuş deney dosyalarını değiştirmez.

Sonrasında, yalnız eksik tutarlı ölçüm varsa mevcut tahminlerden MAE/MAPE
puan kartı; farklı kohortlar ayrı tablo. OOS'tan yeni kalibrasyon/eşik seçimi,
Cotcast karşısında eşleşmemiş doğruluk iddiası yok. Cotcast'in hedef/veri/saat/
test kapsamı bu incelemede doğrulanmadı. MAPE ikincil betimleyici ölçüdür;
%5 MAE /%53–55 yön ve orijinal dönem kapıları korunur.

Yeni bilgi hipotezi elemesi bundan **sonra**, en fazla 2–3 aday ve sıfır fit
ile yapılabilir; ekonomik mekanizma, önceki tam-kapsam deneyi, erişim/vintage,
bağımsız olay sayısı, ortak kontrol, sızıntı riski ve stop koşulu gerekir.
Beklenti verisi olmadan rapor revizyonuna sürpriz denmez; CT=F'den sahte ikinci
kontrat üretilmez. FAS karantinası ve diğer kaynak kabul sınırları açılmaz.
Önce mevcut/açık veri; ücretli veri, GPU veya yeni bağımlılık önerisi yok.

PR zincirinin #10'dan başlayan entegrasyon incelemesi ayrı mühendislik hattıdır;
otomatik merge veya canlı görev/model güncellemesi yok. Gece operasyonunda
mevcut ret nedeni salt okunur incelenebilir; saat/backfill/güç politikası
sessizce değiştirilemez. Bu işler bilimsel adayı başarıya çevirmiyor.
