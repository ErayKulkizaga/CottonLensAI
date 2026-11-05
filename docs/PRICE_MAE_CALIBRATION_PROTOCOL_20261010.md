# Kayıtlı Ridge fiyat-MAE kalibrasyonu — sabit protokol

Profil `price-mae-calibration-v1`. Soru: mevcut Ridge çıktısındaki **global
çarpan yanlılığı**, olgun geçmişte exact fiyat-MAE minimizasyonuyla düzeltilince
aynı OOS origin'lerde yararlı katkı sağlıyor mu? Bu, tam koşullu quantile model
testi veya bütün target/loss farklarının sınanması değildir.

Önceki full-year T+1 artifact'leri checksum ile yeniden hesaplandı:
geçmiş log-getiri medyanı baseline'ı aynı2.006 origin'de Naive kazancı
−%0,0217506,2/8 yıl pozitiftir. Bu baseline tekrar eğitilmeyecek.
Sicilde drift kapsamı kısmidir; eşleşme yokluğu yenilik kanıtı sayılmaz.
Yeni işlem, kayıtlı tahminin **fiyat-ağırlıklı artık medyanıyla** düzeltilmesidir.

## Sabitler

- Kaynak: tamamlanmış Texas `numeric_D0`, published
  `nass-regional-result-20261009`; aynı37 özellik/eksiklikler, fiyatlar,
  T+1 etiketler,2.006 origin,2016–2023.2024+ dışarıda kalır.
- Ridge alpha1/window1/seed42, mevcut scaler/coef/target dönüşümleri aynen
  yüklenir. **Yeni Ridge/model fit'i0**; aynı72 inner+97 outer modelin
  eğitim kohortlarından **169 kapalı-form skaler kalibrasyon güncellemesi**.
  Kalibrasyonun parametre tahmini olduğu saklanmaz veya "hiç öğrenme yok"
  diye sunulmaz. Yeni aile/grid, veri veya model yayını yoktur.
- Expanding history, aynı5 gözlemlik label maturity,21 gözlemde refit;
  geçmiş3×63 blokta aynı `[0,.25,.5,.75,1]` küçültme seçimi. Her kolun seçimi
  kendi geçmiş tahminlerinden yapılır; başka hyperparameter seçilmez.

Her modelin eğitim satırlarında `A_i=C_i*exp(r_i)` ve
`P_i=C_i*exp(model_return_i)` olsun. `sum |A_i-f*P_i|` minimizer'ı,
`A_i/P_i` oranlarının `P_i` ağırlıklı medyanıdır. Eşitlikte küçük minimizer.
Kalibre edilen ham getiri `model_return + log(f)`; sonra geçmişte seçilen
küçültme ağırlığı uygulanır. Faktör her refit'te yalnız o modelin **eğitim**
etiketlerinden hesaplanır; inner/evaluation etiketleri faktörü belirlemez.
Bu in-sample artık kalibrasyonudur; çapraz fit veya conditional quantile değildir.

Olgun etiket, artifact kimliği, baseline/candidate date-target-value eşitliği
ve eski model çıkarımı doğrulanmadan sonuç üretilmez. Uyumsuzluğu gidermek
için sessiz kesişim, origin silme, fallback veya veri düzeltme yapılmaz.
Kaynak saati/ilk vintage varsayımları eski deneyle aynıdır; yeni PIT kabulü yok.

## Ölçüm ve sabit karar

Birincil: **seçilmiş kalibre aday–eski seçilmiş numeric_D0 fiyat-MAE farkı**.
Naive'ye göre kazanç, ham kollar, aktif/tüm-origin yön, yıllar ve en büyük21
Naive hatası çıkarılmış duyarlılık ayrıca verilir. Aynı yıllar içinde paired
bootstrap blok20/60,10.000 tekrar,seed42. Orijinal ağırlıklar yeniden kurulur;
sayısal eşitlikler yıl kazanımı sayılmaz (1e-10 yüzde puan tolerans).

- >=%5 Naive kazancı,>=%53 tüm-origin yön,>=6/8 yıl ve iki blokta control
  karşısında pozitif alt sınır: STRONG_FIXED_RECIPE_CANDIDATE; bağımsız ileri
  doğrulama gerekir, otomatik model yayını yine kapalıdır.
- Control karşısında iki alt sınır pozitif ama pratik eşikler sağlanmıyor:
  SMALL_CALIBRATION_CONTRIBUTION; hedef çözülmüş sayılmaz.
- Pozitif katkı gösterilemiyor ve iki blokta Naive kazancı üst sınırı <%5:
  CALIBRATION_DOES_NOT_RESCUE_FIXED_RECIPE; bu çarpan tarifini büyütme.
- Diğer durum: INCONCLUSIVE; otomatik ek fit yok.

Aralıklar tekrar kullanılmış tarihe ve sabit tarife koşulludur; bütün
araştırma seçimlerini kapsayan bağımsız başarı kanıtı değildir. Sıfıra
küçültülmüş adayın degenerate aralığı kaynakta sıfır sinyal kanıtı değildir.

## Yürütme

Kaynak/test/protokol önce GitHub'a gönderilir; sonra aynı kimlikte bir CPU
süreç/iki thread, mevcut CPU ortamı ve açık
`COTTONLENS_ALLOW_LOCAL_CPU_TABULAR=1` ile çalışır. Güncelleme bütçesi169;
model eğitim fonksiyonu çağrılmaz. `ml/price_mae_calibration.py` açık
`--experiment`, `--reference-manifest`, `--output` ister. Girdiler orijinal
Release envanterine bağlıdır; output eski deney içine yazılamaz. Kaynak,
parametre, her çarpan/kohort, tarih/gerçekleşen hedef, ham/seçilmiş tahmin,
eski kaynak saati metadata'sı ve paired sonuçlar korunur.

Dokuz dar testte weighted-median çözümü bağımsız linear-program minimumuyla,
bilinen yanlılık/sıfır tahmin ve olgunlaşmamış hedef reddiyle sınandı.
Sonuç kendi model/kalibrasyon/sicil kimliğiyle yayımlanır; eski kanıt değişmez.
