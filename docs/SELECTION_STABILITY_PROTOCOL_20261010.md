# İç doğrulama seçim kararlılığı — sıfır-fit tanı

Profil: `weak-signal-selection-stability-v1`. Girdi, tamamlanmış zayıf sinyal
Release'inin checksum doğrulanmış kopyasıdır. On seed × null/enjekte × sekiz
yılın **160 geçmiş seçimi**, yalnız kayıtlı **1.440 iç fit** tahminiyle incelenir.
Yeni fit, veri indirme, hedef değişimi veya yeni ağırlık kümesi yoktur.

Her seçimde önceki 3×63 doğrulama günü ve `[0,.25,.5,.75,1]` fiyat-MAE skorları
yeniden kurulur. Kayıtlı karar ve etiket olgunlaşması eşleşmeden analiz ilerlemez.
Receipt kimliği aynı geçmiş eğitim kohortundan hesaplanır; outer receipt,
outer-predictions CSV ve gerçekleşen OOS hataları seçim analizine girmez.

Çıktılar: en iyi/ikinci skor farkı, her tek bloğun seçimi, üç leave-one-block-out
seçimi ve kayıtlı ağırlığın yeniden seçilme sıklığı. Yalnız geçmiş blokların
içinde paired moving-block bootstrap, blok20/60,10.000 tekrar,seed42 kullanılır.
Her draw'da orijinal mean-of-three-normalized-MAE tanımı korunur; Naive paydası
aynı paired örnekten hesaplanır. Eşitlikte küçük ağırlık seçilir.

Skor farkı aralıkları recentered basic tanıdır; kazananın seçilmiş olmasına
göre düzeltilmiş anlamlılık veya başarı olasılığı değildir. Kayıtlı tahminleri
örneklemek, modeli yeniden eğitmenin belirsizliğini ölçmez.63 günlük blokta
60 günlük parçaların az sayıda başlangıcı olması yapay kararlılık verebilir;
20/60 hassasiyeti formal güç kanıtı değildir. Yeniden görülen tarihçe bağımsız
holdout yapılmaz. Bu tanı sonrası eski RAW_ONLY_RECOVERS kararı değiştirilmez.

Amaç küçültmenin zararını varsaymak değil, küçük gözlenen kaybı seçim
kararsızlığıyla açıklayıp açıklayamadığımızı ölçmektir. Sonuçlar tek sonraki
karara bağlanır; OOS'tan ağırlık seçme veya otomatik piyasa eğitimi yapılmaz.

Komut (mevcut CPU ortamı; OMP/OPENBLAS/MKL thread sınırı2):

```powershell
python ml/selection_stability.py --study EXTRACTED/study --release-manifest research/evidence/weak-signal-result-release-20261009.json --output NEW_ANALYSIS_DIRECTORY
```

Girdi envanteri, kullanılan her dosya, analiz kaynağı ve parametre kimliği
saklanır. Çıktı özgün study içine yazılamaz; değişen kimlikle cache kabul edilmez.
