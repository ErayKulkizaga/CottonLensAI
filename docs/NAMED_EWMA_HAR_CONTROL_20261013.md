# Named T+1: EWMA üzerine sabit kapasitede HAR düzeltmesi

22 piyasa +1 sentetik fit tamamlandı. Üç relatif HAR girdi
`own_har_k-log(origin_named_EWMA)`; eğitim hedefi
`named_actual/origin_named_EWMA`. Dört parametreli QLIKE log-link learner
mutlak varyansa `origin_named_EWMA*exp(predicted_log_ratio)` ile döner.
Sıfır katsayı EWMA'yı verir; yeni kaynak, grid veya kapasite artışı yok.
Önceki standalone QLIKE, OLS ve scale modelleri yeniden fit edilmedi.

| Pozitif değer yeni model lehine | Delta QLIKE | CI20 | CI60 | Top1% trim |
|---|---:|---|---|---:|
| Ham / EWMA | −0,01980127 | [−0,04296094;0,00346456] | [−0,03569407;−0,00349093] | −0,03674297 |
| Past-selected / EWMA | −0,00308231 | [−0,00947628;0,00330095] | [−0,00759651;0,00148791] | −0,00790472 |

Geçmiş üç63validation ağırlık0,25 seçti; dış dönem bunu değiştirmedi.
Ham modelin blok60 aralığı negatiftir; seçilmiş sonuç INCONCLUSIVE.
Bu sabit learner referansı geçmedi. Mevcut EWMA korunur; yeni offset,
coefficient/optimizer/history grid'i veya daha büyük model başarısı çıkarılmaz.

Aynı246ordinal slot/243forecast/242paired,500minimum train,
21gözlem refit ve beş-gözlem maturity korundu. 2023 görülmüş tek yıldır;
bağımsız holdout/6-of-8/live gate değerlendirilmez. Sparse own-history
span236/latestage192, tarihsel UTC/vintage varsayımları değişmedi.

Kaynak `58dda49660cbc2942a665d60cedf3dafd4a3dfbae7282dbd897a0bf435386094`;
tasarım `50f0d66ee28b11546b508a73256f0119ba583512e9f8385c65a002e34f0f01e6`.
354private ön kayıt dosyası taze geri kurularak doğrulandı. Beş yeni
unit/18ilgili test/Ruff geçti; bilinen relatif ortalama sentetik kontrolü
mutlak varyansı geri kazandı. Bu piyasa başarısı değildir.
Native22model/gradient/prediction yeniden kuruldu; bağımsız726Decimal
loss,242EWMA,2.988feature scalar,dörtCI/22cutoff/16.764train-row geçti.
Raporlama fit0. Sonuçlar `output/named-ewma-har-control-20261013/`.
