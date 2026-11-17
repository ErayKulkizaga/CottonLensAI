# Getiri işareti ve sonraki risk: büyüklük kontrollü tanı

Ön kayıtlı pozitif hipotez, aynı büyüklükte yükseliş şokunun sonraki gün
EWMA'ya relatif risk hatasıyla pozitif ilişkisiydi. Tek gösterge:
`sign(r_t)*r_t^2/EWMA_t`; unsigned `r_t^2/EWMA_t` ve yıl intercept'i
ayrı kontrol edildi. Yıl içi partial covariance, her bootstrap örneğinde
tekrar projection/centering ile hesaplandı. Bunlar ex-post tanı katsayılarıdır;
geçmişte öğrenilmiş tahmin veya strateji değildir.

Ekonomik mekanizma için [Chen/Mu2021](https://discovery.dundee.ac.uk/ws/files/49087226/1_s2.0_S2405851320300167_main.pdf)
incelendi. Çalışma pamuğu açıkça dışlar; diğer emtialardaki bulgu CottonLens
becerisi veya pamuktaki işaret için kanıt değildir. Mevcut HAR/symmetric
GARCH getiri işaretini kullanmaz. Sicilde asymmetric sorgusu incelendi;
eşleşme yokluğu tek başına yenilik ispatı sayılmaz.

CT proxy T+1/EWMA0,94, 2016–2023 ve 2.006 ordinal origin değişmez.
20/60 moving-block,10.000draw/seed42,6/8pozitif yıl ve pozitif trim kapısı
hesap öncesi donduruldu. Şok büyüklüğü etkisi işaret bilgisi diye sayılmadı.

| Kontrol | Partial covariance |
|---|---:|
| Nokta | −0,36789234 |
| CI20 | [−0,69250770; −0,01267305] |
| CI60 | [−0,70317441; −0,09480553] |
| Pozitif/negatif yıl | 3/5 |
| Top1% baseline-loss trim | −0,03922468 |

Pozitif inverse-leverage screen **başarısız**. İki aralıkta işaret tersidir;
bu pozitif hipoteze karşı kanıttır. İşareti sonradan değiştirip screen geçti
denmez; negatif yıl5/8dir. Bu covariance değerleri QLIKE kazancı değildir.
Ölçülmüş EWMA üstünlüğü/Colab gerekçesi henüz yoktur. Source clock/vintage
varsayımları ve araştırma geçmişinin yeniden kullanılması korunur;
multiplicity düzeltilmemiştir. Bu kayıtta training_allowed=false/new_fits=0.

Bağımsız denetim current/target exact Cotton endpoint'lerinden Decimal ile
6.018 feature/error scalar kurdu. Bootstrap partialcov alternatif
`lstsq` residual+`numpy.cov` yöntemiyle iki aralık ve trim için doğrulandı.
Bu nuisance projection'lar forecast-model fit'i değildir. Getiri/close
uyuşma max farkı1,39e−17. Üç yeni test unsigned risk'in yanlış asymmetry
üretmediğini, bilinen ilişkinin/missing ordinal'ın korunduğunu ve degenerate
control'un negative score'a çevrilmediğini doğruladı; Ruff temiz.

Yerel kanıt: `output/asymmetric-risk-moment-20261013/v1/`.
Bu inceleme eski model, tahmin, fiyat veya live126 kilidini değiştirmedi.

Private `asymmetric-risk-preflight-20261013` Release taze indirildi;10dosya
checksum ile geri kuruldu.6.018Decimal scalar/iki bağımsızCI ve trim replay
geçti; verification JSON byte eşit. Eski199aşama/541fit tarifi korunur.
Kanıt: `research/evidence/asymmetric-risk-preflight-20261013.json`.
