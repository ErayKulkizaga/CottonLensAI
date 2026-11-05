# Named bilgi kapsamı: sıfır-fit sentez ve unutulan eski fiyatlar

## Karar

**ROOT CAUSE NOT YET IDENTIFIABLE.** Kalan iki açıklama: korumalı saat yeni
konumlanma bilgisini tahmin anından gereksiz yere uzaklaştırıyor; veya bu
pozisyon seviyeleri/temsil, bu sabit modelde pratik T+5 koşullu-ortalama katkısı
taşımıyor. Daha büyük model gerektiği gösterilmedi. Önce tek zamanlama
karşılaştırması ayırıcıdır; model/lag grid'i açılmaz.

Mevcut başarısızlığın tamamını “CT=F kötü”, “bias”, “kaynak modele ulaşmıyor”
veya “bootstrap bozuk” açıklamalarına yüklemek aşağıdaki kanıtla desteklenmiyor.
Sonuçlar tekrar incelenmiş2023 içindir; tarihsel UTC/ilk-vintage belirsizliği sürer.
Yeni fit0; eski veri, native modeller, deney kimlikleri ve eşikler değiştirilmedi.

## Sonuçların gerçekten ayırdığı şey

| Tamamlanmış kontrol | Ham katkı, Naive-MAE yüzde puanı | Kanıt kapsamı |
|---|---:|---|
| CT etiketi yerine named etiket | +13,378639 | Aynı hedefte label transport farkı; bütün eski CT araştırması yanlış demek değil |
| Named Ridge yerine sabit XGB | +7,444570 | Kapasite bu eşleştirmede önemli; XGB Naive kazancı hâlâ−%0,493740 |
| Dört aynı-kontrat getirisi, numeric–mask | −2,512024 | Kaynak işareti belirsiz; dört alan eklendi, CT teknikleri çıkarılmadı |
| Korumalı CFTC L1, numeric–mask | +1,809270 | Kaynak işareti belirsiz; numeric Naive−%1,124208, hedef sağlanmadı |
| Üç bilinen zayıf öğretici | Beklenen ham%8,38/%8,64/%8,50 | Basit yararlı etki öğrenilebilir; piyasa becerisi veya tüm temsil/güç kanıtı değil |

## VERIFIED — modeller gerçekten hangi bilgiyi kullanıyor?

CT=F ve aynı kontratın geçmiş getirilerinde242eşleşmiş origin:

| Geçmiş pencere | Korelasyon | Yön uyumu | Medyan mutlak fark, bp |
|---|---:|---:|---:|
|1|0,943989|%92,561983|15,762986|
|5|0,914604|%90,909091|51,348906|
|10|0,905217|%86,363636|89,236037|
|20|0,903153|%88,429752|107,998363|

Hiçbiri `1e-12` içinde eşit değil; fakat kuvvetle ilişkili. Bu korelasyonlar
**OOS R² değildir** ve fiyat proxy'sinin bütün teknik alanlarının yararlı
olduğunu kanıtlamaz. CT tekniklerini tutmanın zarar verdiği de bu sayılardan
çıkarılamaz. Aynı-kontrat geçmişi eklenmiş; tam teknik blok değiştirilmemiştir.

CFTC numeric kolun12dış refit modelinin **12'si de üç sayısal pozisyon alanını
kullanıyor:1.032 split**, toplam3.282split'in%31,44'ü. CT'den türetilen
alanlar1.395split. Mask kolunda CFTC'nin yaş/eksiklik bilgisi yalnız9split.
Kaynak değerleri scaler'da kaybolmamış veya modelden dışlanmış değil.
Split sayısı, OOS fayda ya da nedensel önem değildir; modelin kaynağı kullanıp
genellenemeyen ilişki öğrenmesi de aynı görünümü üretir.

## VERIFIED — sabit bias düzeltmesi hedefi kurtarmıyor

Gelecek gerçek fiyatları kullanarak bulunan en iyi sabit log-offset,
fiyat-MAE'nin weighted-median optimumudur. Bu bir **hindsight üst sınır**;
uygulanabilir kalibrasyon, yeni model veya bütün politikaların sınırı değildir.

| Saklanan ham tahmin | Naive kazancı | Hindsight offset sonrası kazanç |
|---|---:|---:|
|Own-history mask / eski named XGB karşılığı|−%0,493740|%2,445597|
|Own-history numeric|−%3,005765|%0,595311|
|CFTC mask|−%2,933479|%0,647873|
|CFTC numeric|−%1,124208|−%0,059724|

Hiçbirinde%5 yok. Yeni bias grid'i gerekçelenmedi. Önceki T+1 global
kalibrasyon başarısızlığı da korunur; tam koşullu quantile modelleri bu
üst sınırla reddedilmiş sayılmaz.

## VERIFIED — belirsizlik kontrolünün sınırı

Aynı246slot/242uygun satırda, aynı10.000/seed42/blok20–60 bootstrap saklanan
pozitif öğreticilere uygulandı. Seed2703'ün gerçek beklenen kazancı%8,500825;
gerçekleşmiş kazanç%1,021117. Aralıklar20:[−%3,775359,%6,094834],
60:[−%2,138409,%4,767808]; bilinen beklenen etki ikisinde de dışarıda.
Diğer iki öğreticide beklenti aralıkların içinde.

**Bu üç örnek kalibre coverage veya piyasa gücü ölçümü değildir.** Bir
beklentinin95%aralık dışında kalması tek başına uygulama hatası kanıtlamaz.
İki üst sınırı birlikte%5altında gerektiren mevcut pratik karar kuralı,
üç ham öğreticinin hiçbirini yanlış “hedef dışlandı” diye sınıflandırmıyor.
Eşikler değiştirilmez. Nonsignificance veya küçültülmüş `[0,0]` farkı,
kaynakta sinyal yokluğu olarak yorumlanmaz.

## VERIFIED — kaynak kapsamı düzeltilmesi

997quote panelinin2020başlangıcı bütün yerel arşivde eski fiyat yokluğu
değilmiş. İncelenen üç mevcut önbellekte60kopya/**52benzersiz**2018–2019
raporu bulundu: **520futures fiyatı**,2018-11-21–2019-06-03. Mevcut
`price_reconciliation.parse_quotes` bunları zaten okuyor; hyphen/date parser
hatası iddiası yanlış. Bağımsız token/Decimal okuyucusu520fiyatı eşleştirdi.
Sorun bu legacy ESMIS metinlerinin2020+MMN derlenmiş quote kapsamına alınmaması.
Eski “2016–2019 yok” tablosu kendi997girdisi içindir; bütün mevcut dosyalar
hakkında kullanılmaz. Eski signed kanıtlar değiştirilmedi.

52tarihin51'inde ilk quote mevcut CT fiyatıyla float32olarak aynı.2018-11-26
76,75versus77,4000015259farkı saklandı; diğer quote ile eşleşmiyor. Exchange
roll veya bozulma kanıtı değildir; fiyat-alanı farkı henüz açıklanmadı.
Sekiz duplicate aynı report byte/quote değerlerine sahip farklı URI'lerdir;
bütün page/receipt sürümleri korunur, ilk-vintage seçilmiş sayılmaz.

Eski997row aynen korundu; ayrı **1.049row candidate union** oluşturuldu.
Sabit mevcut label algebra'sıyla39ek T+5etiketi kuruldu:969→**1.008**;
2020–2023origin/hedef fiyatları, kontratlar, uygunluk ve eski return bitleri
aynı. Log'un yeniden hesaplanmasıyla oluşan `9,93e-17` CSV okuyucu kuyruğu
ölçüldü; eski label bitleri korunarak adayda açıkça uzlaştırıldı. Bu ölçek
tahmin başarısızlığını açıklamaz. Yeni okuyucular round-trip hassasiyet kullanmalı.

Bu **aday veri ekidir**, training-ready veya admitted dataset değildir.
Eski sayfaların saat/ilk-vintage/Final statüsü sertifikalanmadı. Kaynak
ingestion zamanı yayın zamanı olarak kullanılmaz. Yeni veri indirme yok;
quarantine kardeş klasörleri otomatik alınmadı.52ham belge/page/receipt
ve ayrı label CSV private kanıt paketinde korunur; eski history yazılmaz.

## VERIFIED — formal kanıt kapasitesi

Mevcut3520Cotton satırında yalnız969named T+5etiketi vardı. Ek39ile:

| Değerlendirme yılı | En erken olgun train, mevcut | Aday ekle |
|---|---:|---:|
|2021|32|71|
|2022|279|318|
|2023|528|567|

500train+geçmiş189validation kuralıyla **hâlâ yalnız2023** destekleniyor.
2016–2019ham Cotton geçmişi, named fiyat etiketi demek değildir. Bu açık
kapsam6/8kapısını sağlayamaz; daha önce belgelenmiş sınır koddan yeniden
doğrulandı. 500ve6/8değiştirilmez,2024+seçim için kullanılmaz.52ek raporun
geri kazanılması yeni değerlendirme yılı veya pratik skill üretmedi.

## PLAUSIBLE — kalan zamanlama açıklaması ve tek ayırıcı test

Mevcut CFTC source:ref+7takvim günü, **sonraki**Cotton origin. Normal
COT yayın saati olarak Cuma15:30NewYork kullanılması bir **varsayım**;
[CFTC'nin genel takvimi](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
bütün tarihsel haftaları veya ilk sürümü kanıtlamaz. Bu karşı-olgusal saat
246origin'in126'sında seçilen raporu değiştiriyor. Federal/GoodFriday/
ThanksgivingFriday/Dec24/Dec31haftaları ve non-Tuesday referanslarda eski
korumalı saati tutan belirsizlik guard'ıyla bile **97origin**,39,43%değişiyor.
Bu guard gerçek CFTC takvim sertifikası değildir. Gözlenen timing katkısı yok;
mevcut modeller bu yeni saatle çalışmadı.

**Tek sonraki fit önerisi:** aynı37alanlı namedT+5/sabitXGB, yalnız korumalı
ordinary-week kaynak saatini değiştir; mevcut CFTC numeric benchmark'i
saklanan native modellerden yeniden kullan. Aynı eski997quote paneli/etiket/
origin/train/3×63/refit/shrinkage/baseline; 52rapor/39etiket bu teste karışmaz.
Bir yeni kol **22CPUmarket fit**, eski dört aynı-recipe mekanik kontrolün
CORE/recipe bağları ve native replay ile referansı; yeniden132sentetik yok.
Yeni kaynak/data/ortam/clock/rapor kimliği ve taze GitHub ön kayıt olmadan
fit başlamaz. Yeni model, lag veya parametre araması yok.

Birincil ham paired price-MAE kazancı mevcut numeric benchmark'a karşı;
aynı20/60bootstrap, Naive%5/%55ikincil pratik sınır, aynı tam cohort.
Güçlü pozitif fark gecikme açıklamasını destekler, fakat ancak saat/vintage
varsayımı içinde; gerçek PIT veya6/8başarısı değildir. Etki yoksa bu kaynak/
temsil/modelde korumalı gecikmeyi kaldırmak hedefi kurtarmıyor; başka grid
otomatik başlamaz. Kaynakta bütün sinyal yokluğu sonucu çıkarılmaz.
