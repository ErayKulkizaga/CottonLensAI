# Aynı kapasitede own-level ve DXY-risk kontrolü — 29 Ekim 2026

Önceki dört parametreli risk tarifinde EWMA katsayısı `1-sum(HAR slopes)`
olmak zorundadır. Bu kısıt doğrulandı; gerçek piyasa başarısızlığını
açıkladığı henüz gösterilmedi. Mevcut DXY fiyat deneyleri de DXY risk
bilgisinin katkısını sınamış sayılmaz. Bu deney iki açıklamayı ayırır.

## Fit öncesi kilitli tasarım

`risk-level-source-t1-v1`; aynı 2016–2023 dönemindeki 2.006 ordinal tarih,
1.998 kullanılabilir forecast ve 1.989 eşlenmiş hedef. Her iki kol beş
parametrelidir. A, aynı üç relative-HAR'a serbest `log(origin named EWMA)`;
B, aynı üç relative-HAR'a `log(DXY_RV5)` ekler. Her ikisi
`EWMA * exp(intercept + four predictors)` üretir. DXY alanından EWMA
çıkarılmaz; böylece eksik own-level katsayısı dış kaynak diye gizlenmez.

RV5, ham DXY gözlem takviminde ardışık beş log-getiri karesinin ortalaması;
sonra son kullanılabilir state seçilir. Varsayım:
`source date + 1 day 00:00 UTC <= origin + 1 day 00:15 UTC`.
Sınır eşitliği dahildir, azami yaş üç Cotton gözlemidir; veri boşluğu veya
geçersiz fiyat atlanarak geçmiş pencere sıkıştırılmaz. Log için mevcut
1e-12 tabanı kullanılır. Bu saat/ilk-vintage doğrulaması veya timing-only
ablation değildir. İki kolun aynı kapasitede kıyası, DXY'nin unrestricted
own-history modeline koşullu bütün artı bilgisini de tek başına kanıtlamaz.

174 eski native kontrol fit'i yeniden kullanılacak. Her yeni kol aynı174
işi, geçmiş3×63 validation ve21refit takvimini kullanır. Olgunluk beş seans,
expanding train ve ağırlıklar0/.25/.5/.75/1 değişmez. Aynı OLS+smearing
başlangıcı, train-only normalization, BFGS max500/gtol1e-6; fallback yok.
Bütçe348 piyasa +1 ortak kernel sentetiği; tek süreç/iki thread ve30dakika
checkpoint oturumu. Eski deneyler ve veriler değiştirilmez.

Birincil karşılaştırma ham DXY eksi ham own-level kaybının ters işaretlisi:
`QLIKE(own-level)-QLIKE(DXY)`; pozitif DXY lehinedir. Her ham kolun EWMA ve
eski dört parametreliye katkısı ayrıca raporlanır. Geçmişte seçilen kollar
ikincildir; dış sonuçla kazanan seçilip yayınlanmaz. Yılları aşmayan paired
bootstrap10.000/seed42/blok20ve60;≥6/8pozitif yıl ve en büyük%1EWMA-kaybı
çıkarıldığında pozitif katkı kapıları değişmez. Negatif QLIKE'den yüzde
başarı hesaplanmaz. İncelenmiş tarihsel yıllar bağımsız holdout değildir.

Ham DXY QA: 2.006 origin eksiksiz/pozitif state,2.005'inde source date aynı
origin, azami yaş1; ham kaynak SHA23cfabb7… önceki doğrulanmış arşivdir.
Sekiz özgün no-forecast tarih korunur. Yeni veri indirme, GPU, grid,
collector veya otomatik model yayını yoktur.

Kod ve veri önce private Release'e alınacak; taze indirilen snapshot ile
`verify` geçmeden fit açılmayacak. Sonuçlar native model/selection, bağımsız
Decimal-kayıp/CI replay ve taze restore ile korunacak. Bu belge fit öncesi
tasarımı kaydeder; sonuç henüz yoktur.
