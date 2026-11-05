# T+1 named risk: sabit kapasitede eğitim loss kontrolü

22 yeni piyasa ve bir sentetik fit tamamlandı. Önceki 22 own-history
OLS+smearing fit'i native kaynak/ortam/tahmin eşliğiyle yeniden kullanıldı;
yeniden eğitilmedi. Yalnız eğitim criterion'u QLIKE oldu. Dört parametre,
aynı üç HAR özelliği, named hedef, train/validation tarihleri, EWMA0,94,
21-gözlem refit ve 246 ordinal slot korundu. 243 tahmin/242 paired sonuç;
eksik slotlar silinmedi. Kaynak `1b71befe6e12fc83efda83a0d6bac2c72e781f7add137cf9cdf203e9c48cb7e9`.

| Pozitif değer aday lehine | Delta QLIKE | CI20 | CI60 |
|---|---:|---|---|
| QLIKE learner / eski OLS learner | −0,00923334 | [−0,02448453;0,00872237] | [−0,02292606;0,00933961] |
| Ham QLIKE learner / EWMA | −0,01862219 | [−0,04971762;0,01256208] | [−0,04272736;0,00524323] |
| Seçilmiş QLIKE learner / EWMA | 0 | [0;0] | [0;0] |

Geçmiş validation QLIKE learner ağırlığını0 seçti. Eşitlik referansa
dönüştür; model becerisi veya kaynakta sıfır sinyal kanıtı değildir.
En büyük%1 EWMA kaybı çıkarılınca ham katkı−0,04222947. Loss mekanizması
INCONCLUSIVE; pozitif başarı gösterilmedi. Bu sabit loss kolu kapanır;
optimizer/grid/seed araması veya Colab kapasite artışı gerekçesi yok.

22 optimizer native yeniden kuruldu: gradient∞≤7,52e−7, her fit'te
training objective başlangıçtan daha iyi. Sentetik bilinen ortalama
kontrolü geçer; sentetik sonuç piyasa başarısı değildir. 42 ilgili test
geçti. Bağımsız Decimal kontrol242 aynı-kontrat EWMA,1.210loss,
2.988HAR feature scalar,22 maturity cutoff/16.764train-row ve sekiz CI
doğruladı; raporlama sırasında fit0.

2023 görülmüş tek yıldır; bağımsız holdout/6-of-8 başarı değildir.
Observed-pair pencerenin train span236/latestage192 sınırı değişmedi.
Tarihsel UTC/yayın/vintage hâlâ varsayım; fiyat/yön kapıları korunur.

Yeni sıfır-fit tanı: güçlü referansın EWMA state'i üç HAR girdisinde
bulunmuyor; past log-EWMA'nın HAR'a lineer projeksiyon artık sd'si tüm
joblarda≥0,13879. Aynı olgun train'de mean(actual/EWMA)1,16375–1,24659.
Bu geçmiş ölçek koşulu ayrı EWMA kalibrasyon sorusunu gerekçelendirir;
OOS sinyal veya yeni model sonucu değildir. Önce geçmiş sicilde eşdeğer
kontrol aranmalı, ardından tek sabit analytik past-only ölçek kontrolü
ayrı source/design/budget kimliğiyle hazırlanmalıdır. Eski HAR global
smearing kalibrasyonuyla EWMA kalibrasyonu eşdeğer değildir.

Kanıt: `output/named-risk-loss-control-20261012/execution-v1/reports/`,
`independent-verification.json`, `baseline-information-diagnosis.json`.
Eski fiyatlar, etiketler, modeller, canlı collector ve126-origin kilidi korunur.
