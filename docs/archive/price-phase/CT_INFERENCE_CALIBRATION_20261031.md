# İleri risk kararının sentetik kalibrasyon kontrolü

Profil `ct-forward-inference-calibration-v1`; **sıfır yeni fit, sıfır yeni
dataset, sıfır ileri performans skoru**. Aktif 252-origin çalışma, model,
eşikler ve saatler değişmedi. Bu kontrol piyasa başarısı değildir.

## Soru ve sabit tasarım

Önceki koşullu normal-HAC hesabı, 252 gözlemde tarihsel +0,01770104
etkisi için tek aralıkta yaklaşık %14–15 saptama veriyordu. İki aralığın
ortak kararını veya gerçek piyasa gücünü ölçmüyordu; o analiz tekrarlanmadı.

Yeni kontrol, mevcut `difference_interval` fonksiyonunun **aynı** 10.000
tekrar/seed42/blok20 ve60 kararını kullanır. Gerçek 43+209 yıllık bölünme,
pooled252 yalnız duyarlılık; her koşulda 1.024 bağımsız sentetik seri.
Seri/koşullar aynı yenilikleri paylaşır: sekiz koşul, 8.192 bağımsız
piyasa deneyi değildir. Sabit seed20261031; AR katsayısı yalnız 0 veya0,8.

Gerçekleşen kare getiri vekili, pozitif iki sabit varyans tahmini arasında
beklenen QLIKE farkı tam sıfır veya +0,01770104 olacak şekilde üretildi.
Bu dağılımlar piyasaya fit edilmedi; gerçek piyasa gücü tahmin edilmedi.
Oranlar 252 yayının tamamının zamanında bulunduğu koşula aittir;
bilgisayar/kaynak kesintilerinin operasyonel etkisini kapsamaz.
Tüm koşullar, yorum ve bütçe sonuçtan önce kaydedildi. Kaynak
`7a2e2018…`, private 294 dosyalık ön kayıt taze geri kurulduktan sonra
çalıştı. İlk Release oluşturma isteğinin kısa commit hatası korundu;
aynı capture tam commit ile teslim edildi, analiz erken başlatılmadı.

## Sonuç ve karşı kanıt

| Gerçek takvim / sentetik koşul | Ortak pozitif karar | Monte Carlo %95 Wilson aralığı |
|---|---:|---:|
|Sıfır etki, bağımsız kare-Gauss|21/1.024 = %2,05|%1,35–3,11|
|Sıfır etki, AR0,8 kare-Gauss|18/1.024 = %1,76|%1,11–2,76|
|+0,01770104 etki, bağımsız|118/1.024 = %11,52|%9,71–13,62|
|+0,01770104 etki, AR0,8|48/1.024 = %4,69|%3,55–6,16|

**VERIFIED:** 60'lık blok 43 gözlemlik bölümü sabit alır; o bölümden
resampling belirsizliği üretmez. IID varsayımında sonsuz-bootstrap
varyansının gerçek örneklem-ortalama varyansına beklenen oranı blok20'de
0,82066, blok60'ta 0,56092'dir. Bu oranlar cebirsel olarak ve bağımsız
kesir hesabıyla doğrulandı; gerçek piyasa varyansı olarak yorumlanmaz.

**Karşı kanıt:** Tek60-aralık sıfır-etki koşulunda 74/1.024 pozitifken,
iki aralık birlikte yalnız21/1.024 pozitiftir. Ön kayıtlı birincil ortak
kararda aşırı yanlış-pozitif **gösterilmedi**. Wilson aralığı %2,5 sınırını
içeriyor; bu da tüm dağılımlarda kalibrasyonun kanıtı değildir. Kısa yıl
şüphesi tek başına aktif ortak kararın bozuk olduğu sonucunu desteklemez.

**Koşullu bulgu:** Küçük bir gerçek ortalama farkın 252 gözlemde ortak
kararı geçmesi bu iki sentetik dağılımda zordur. Gelecekte başarı
olasılığının %11,5 veya %4,7 olduğu, etki bulunmadığı, daha uzun sürenin
başarı getireceği ya da eşiklerin gevşetilmesi gerektiği sonucu çıkarılamaz.

## Doğrulama, kapsam ve karar

Çalışma 9,21 saniye / tek CPU süreç / iki BLAS thread ile tamamlandı.
24 aralık gerçek mevcut yardımcıyla karşılaştırıldı. Ayrı doğrulayıcı,
AR üretimini `scipy.signal.lfilter`, resampling sayımlarını aralık uçları
ve prefix toplamları, varyans oranlarını tam kesirler ile yeniden kurdu;
2.097.152 sentetik scalar ve sekiz koşul geçti. Doğrulayıcı sonuçtan sonra
yazıldı; ön kayıtlı üretici, tohum, koşullar ve karar değiştirilmedi.

Tam sonuç/private geri kurma ve replay makbuzları sicil ek kanıtındadır.
İlk 10 piyasa fiti tekrarlanmadı; study ve trials byte hash'leri korundu.

**Karar: INCONCLUSIVE — piyasa üstünlüğü açısından.** Aktif ileri test
aynen sürer; 252 seans güç garantisi olarak sunulmaz. Bu kontrol yeni
model/grid, farklı ufuk, ara skor veya test süresi değişikliği açmaz.
Sonraki gerçek bilimsel girdi, 2 Kasım hedefinin 3 Kasım03:05TR kaynak
makbuzudur; dönem ortasında performans skoru çıkarılmaz.
