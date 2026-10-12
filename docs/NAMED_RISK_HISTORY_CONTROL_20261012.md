# T+1 risk: aynı kontratın geçmiş özellikleri

## Karar

**VERIFIED:** aynı named hedefiyle eğitilen sabit Log-HAR'da, CT geçmişi
yerine origin kontratının gözlenen günlük fiyat çiftlerini kullanmak
QLIKE0,02603470 katkı sağladı; iki blok aralığı pozitif. Bu, ön kayıtlı
**gözlenen-çift geçmiş temsili** müdahalesinin bir yıllık koşullu bulgusudur.
Calendar-22 HAR ile named calendar-22 HAR'ın saf karşılaştırması değildir.

**Referansı geçen sağlam risk sinyali yok:** ham own-history HAR'ın
EWMA katkısı−0,00938885; geçmişte seçilen weight0,25 katkısı+0,00266572.
İki seçilmiş aralık sıfırı içerir; en büyük%1 EWMA kaybı çıkarılınca
−0,00483630 olur. Colab/GPU kapasite artışı veya genel başarı ilanı yok.

## Tasarım ve sonuç

Profil `named-risk-history-control-t1-v1`, kaynak
`b2210ea149ee3d7c1daabfef1bbadb067a970a9baeff3e84720f7449e939a7e3`.
2023:246 ordinal slot;243 origin'de özellik/tahmin;242 paired gerçekleşen.
Eksik üç tahmin iki kolda da açık unavailable; impute veya sessiz kesişim yok.
İki kol aynı named etiketi, aynı yeni ortak train/validation tarihlerini,
21 Cotton gözleminde refit, üç63 validation, beş-gözlemlik olgunlaşmayı ve
aynı named EWMA lambda0,94'ü kullanır. En erken train554;500 sınırı korunur.
Önceki label çalışmasındaki576 train ile doğrudan eğitim etkisi toplamı
çıkarılmaz; bu çalışmada iki kol ortak554+ nüfusta yeniden eğitildi.

Müdahale log-mean son1/5/22 **gözlenen**, tam ardışık Cotton-date fiyat
çifti karesidir; kontrat ismi origin'de seçilir ve geçmişte aynı isim aranır.
Eksik kaynaktan çok-günlük değişim günlük getiri gibi üretilmez; eski
bilginin yaşı/pencere takvim genişliği ayrıca korunur. Future veri değişimi
past feature'ı değiştirmez. Varsayımlı archive saat/vintage canlı kanıt değildir.

| Karşılaştırma; pozitif aday lehine | Delta QLIKE | CI20 | CI60 | Top1% EWMA kaybı çıkarılmış |
|---|---:|---|---|---:|
| CT-history → own-history ham HAR | +0,02603470 | [0,00221103;0,04763268] | [0,01053969;0,04268159] | +0,02739442 |
| Own-history ham HAR → EWMA | −0,00938885 | [−0,03881294;0,02157083] | [−0,03215087;0,01650197] | −0,03328392 |
| CT-history ham HAR → EWMA | −0,03542355 | [−0,07906115;0,01063199] | [−0,06824178;−0,00102594] | −0,06067834 |
| Own-history seçilmiş → EWMA | +0,00266572 | [−0,00641059;0,01186980] | [−0,00393418;0,01023191] | −0,00483630 |

Weight CT0, own0,25; yalnız geçmiş QLIKE ile seçildi. QLIKE oran/yüzdeye
çevrilmez.10.000/seed42, full-ordinal blok20/60, eksik slot weight0.
DECISIVE_POSITIVE yalnız ön kayıtlı feature-representation mekanizması
içindir; EWMA üstünlüğü INCONCLUSIVE. Tek görülmüş yıl6/8 veya holdout değildir.

## Yeni bulunan süreklilik sınırı

**VERIFIED:**996 kullanılan train/test origin'in independent kontrolünde
22 gözlenen çiftin pencere genişliği en çok236 Cotton gözlemi; son çift
yaşı en çok192 gözlem. Bu değerler gerçek model train girdilerinde kullanıldı.
Dış2023 origin'lerde en çok24-genişlik ve2-yaş var. Dolayısıyla eski train
ile dış pencere arasında bilgi yoğunluğu farkı mevcut. Bu bir fiyat
bozulması/leakage kanıtı değildir; gözlenen-pair temsilinin önemli sınırıdır.

Sıfır-fit kapsam tanısı: tam calendar22/age0 şartı ilk train'i436'ya,
tanımlayıcı span≤25/age≤3 şartı498'e indirir. İkinci tanı yeni seçilmiş
eşik değildir.500 kuralını498'e düşürmeyiz.20+ gözlenen çift,20+ güncel
günlük gözlem demek değildir; ham label sayısı temiz feature tarihçesi
sayısıyla karıştırılmaz. Strict calendar özelliğini tam2023 diye sessizce
devreye almak aynı protokolü korumuş sayılmaz.

## Hedef/loss ek tanısı

Yeni fit olmadan, native modelin **smearing öncesi latent log tahmini**,
aynı train'de geçmiş kare-getiri logu için kalibre edilmiş log-EWMA ile
eşlendi. Own modelin optimize ettiği log-MSE katkısı−0,03714097;
CI20[-0,11860401;0,03167238],CI60[-0,10239951;0,01087074]. Bu yeni başarı
metriği değildir. `EWMA ham logu` karşısında Jensen farkını başarı diye
sunmak yerine past offset kalibrasyonu uygulandı.

**NOT SUPPORTED:** model zaten optimize ettiği nicelikte üstün ve yalnız
QLIKE bunu saklıyor iddiası. Eğitimde QLIKE slope gradient'i en çok0,11833;
OLS+smearing QLIKE optimumu değildir. Bu tek başına başka objective'nin
OOS üstünlüğünü kanıtlamaz; in-sample gradient piyasa sinyali sayılmaz.

## Doğrulama

44 market fit; yeni synthetic0. Eski HAR native kontrolü üç learner dosyası,
ortam/lock kimliği ve sayısal prediction replay ile tekrar doğrulandı;
eski genel passed cache kabul edilmedi. Maksimum bağıl hata3,33e−15.
40 ilgili test/Ruff geçti. Native44model/recipe/input/selection doğrulaması;
independent Decimal242actual/EWMA,1.210loss,2.988feature scalars/996origin,
sekiz CI,22cutoff/16.764train-row olgunluğu ve15input hash'i geçti.

Eski label deneyleri tekrar edilmedi; source/veri/native kayıtlar korunur.
[Teslimat makbuzu](../research/evidence/named-risk-history-result-20261012.json)
577private Release dosyasını taze indirme/native restore ile doğrular;
result/paired CSV/independent receipt byte-identical, yeni fit0 replay.

**Tek sonraki kontrollü soru:** aynı gözlenen-pair bilgi, aynı dört model
parametresi, aynı named etiket/cohort/reference korunurken yalnız eğitim
criterion'u QLIKE olursa OOS fark var mı?22named refit'te normalized training
gradient har1 için hep pozitif, har22 için21/22pozitif; mean[0,06157;
−0,00777;0,04680]. Intercept gradient en çok3,54e−11; global ölçek düzeltmesi
aynı sorunu yeniden sınamak olur. Bir objective kontrolü henüz uygulanmadı/
eğitilmedi. Slope tanısı bu soruyu gerekçelendirir; kazanç vaadi değildir.
Takvim sürekliliği sınırı, varsa loss kontrolünde de açık kalacak;0 veya498
train'e göre kural değiştirilmez. Daha büyük model veya ufuk taraması yok.
