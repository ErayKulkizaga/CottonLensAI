# Aynı kapasitede own-level ve DXY-risk kontrolü — 29 Ekim 2026

## Sonuç ve daha sonra doğrulanan kaynak kusuru

348piyasa+1sentetik fit tamamlandı;0başarısız market attempt.174eski native
kontrol yeniden eğitilmedi. Aşağıdaki sayılar yeniden üretilebilir fakat
**temiz aynı-kontrat piyasa performansı için METHODOLOGICALLY_COMPROMISED**:

| EWMA'ya katkı | QLIKE farkı | CI20 | CI60 | Pozitif yıl | Üst%1trim |
|---|---:|---|---|---:|---:|
| Own-level ham | +0,00836778 | [-0,01952206;+0,03763683] | [-0,01659487;+0,03471418] |6/8|−0,02128599|
| Own-level past-selected | +0,01268383 | [-0,01015188;+0,03766010] | [-0,00878068;+0,03610564] |5/8|−0,00878068|
| DXY ham | +0,00625212 | [-0,00875641;+0,02130512] | [-0,00667865;+0,01947520] |6/8|−0,01242227|
| DXY past-selected | +0,00429487 | [-0,00589880;+0,01462966] | [-0,00444016;+0,01389175] |5/8|−0,00656082|

Hiçbir üstünlük kapısı geçmedi. DXY hamın eski4parametreliye katkısı
−0,00052277, own-level hamınki+0,00159289; ikisinin aralıkları da0içerir.
Eğitimde ek parametre her174işte kaybı düşürdü: medyan own-level+0,02923,
DXY+0,0001944. Bu tanımlayıcı train/OOS ayrışması yeni model seçimi değildir.

Hata tarihleri denetiminde2020-02-26 origininin2019-07-01 kontratına
bağlandığı görüldü. Ham `2019-2020 Price file.xlsx` dosyası
SHA`03d7347b7a0a5c9b5c277cd39666cb9336deb00d9723d1ff17d8513115311ea6`
`xl/worksheets/sheet4.xml` içinde B1433=43887(2020-02-26),
C1433=43647(2019-07-01), D1433=65.47 içeriyor. Parser hamdaki kimliği
doğru aktarmış; kaynak kabulü, vade yaşam döngüsünü denetlememiş.
[ICE kontrat şartları](https://www.ice.com/products/254/cotton-no-2-futures/specs)
uyarınca geçmiş teslim ayı bu tarihte canlı aynı-kontrat kabul edilemez.

218history/217eval origin,2019-08-01..2020-06-11 arasında bu gerekli
invariant'ı ihlal eder; eval yılları106/111. Ham2018–2019 yıllığı da
tam dönem aynı on kontrat kimliğini taşır;2019–2020 dosyasında aynı liste
2020-06-11'e kadar sürer. İki dosyanın gerçek kontrat/fiyat eşleşmesi
bağımsız günlük kaynakla kurulmadan yıl düzeltmek, eski vadeleri silmek
veya “sinyal yok” demek doğru değildir. Hatanın tüm proje başarısızlığını
açıkladığı henüz gösterilmedi; önceki CT hatları bu kaynağı kullanmaz.

Current prepare/run, geçmiş vade ayını fit'ten önce reddeder. Same-month
son işlem günü için bu gerekli kontrol yeterli değildir; gerçek expiry
takvimi ayrıca gerekir. Eski raw/native/snapshot'lar değiştirilmedi.
Tek sonraki adım sorunlu günlük kaydın gerçek kontrat kimliğini bağımsız
kanıtla bulmaktır; yeni fit veya349fit tekrarı değildir.

Raporun ilk okumadaki plain-model-JSON/hashed-ledger-JSON hatası yalnız
rapor okuyucusunda düzeltildi; fit tekrar edilmedi. Eğitim source5d2863…
ve raporlama source489726… ayrı snapshot'larda korunur. Bağımsız doğrulama
348native/7.020forecast/13.923Decimal loss/16past selection/20CI geçti.
Bu aritmetik doğrulama, hatalı kontrat kimliğini bilimsel olarak geçerli
yapmaz. Sonuç sürümü ve ek kaynak düzeltmesi ayrı korunur.

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
tasarımı kaydeder; bu bölüm fit öncesi tarihsel kayıttır, sonuç yukarıdadır.
