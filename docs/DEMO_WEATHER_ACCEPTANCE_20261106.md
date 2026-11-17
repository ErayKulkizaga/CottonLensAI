# Demo, mevsimsel hava ve ekran gereksinimleri — uygulama kaydı

## Demo gözlem aracı

research/demo_observations.py elle girilmiş gözlemleri mevcut freeze_record ve
writer altyapısıyla değişmez olaylar halinde saklar. Aynı son girdinin tekrarını
çoğaltmaz; revizyonlar ilk gözlemi korur. Geçmiş hedef, gelecekteki gözlem zamanı,
NaN/negatif fiyat, eksik alan ve geri tarihli ekleme reddedilir. Zincir kimliği ve
sırası okunurken denetlenir. Kaydın hash'i yayıncı doğrulaması değildir.

Bu araç tarayıcıya bağlanmaz ve zamanlanmış görev oluşturmaz. Gözlem saati elle
bildirilir; recorded_at sistemin kayıt saatidir. Sonradan girilmiş eski notlar
ilk gözlem için bağımsız zaman tanığı sayılmaz. UNKNOWN değerleri korunur.

Komutlar (depo kökünde PYTHONPATH=ml/src;.):

    python research/demo_observations.py --root output/demo-observations --input observation.json
    python research/demo_observations.py --root output/demo-observations

Girdi alanları: provider, page, observed_at (timezone zorunlu), target_date,
horizon, forecast, unit, contract, price_field, displayed_issue_time, origin_price.
Bilinmeyen son dört alan UNKNOWN olabilir. Farklı ufuklar ayrı serilerdir.
İlk kayıt sonradan UNKNOWN kontratı tahminen tamamlanarak değiştirilmez; yeni
kanıt ve ayrı eşleştirme gerekir. Henüz bağımsız aynı-kontrat gerçekleşmeleri
ve kaynak kabulü olmadığından birincil puanlama her durumda kapalıdır; araç
bilimsel başarı ilan etmez. Sonraki evaluator bu şartları ayrı doğrulamalıdır.

Bu uygulama turunda mevcut IAB oturumunda sekme listesi boş geldi. Yeni canlı
Cotcast gözlemi oluşturulmadı; eski gözlemler geriye dönük forward veri yapılmadı.
Yerel keşif notları output/cotcast-review-20261106 altında korunuyor.

## Mevsimsel hava kaynak araştırması kararı

Karar: DEFER_SOURCE_ADMISSION. Yeni hava ön kaydı veya piyasa eğitimi açılmadı.

Önceki weather-exploration-v1, NASA POWER günlük geçmiş yağış/Tmax, üç sabit
Southern High Plains noktası ve T+5 hedefi kullanır. 7/30 günlük yağış, 7 günlük
sıcaklık ve 35C üstü birikim söz konusudur. Kaynak first-vintage/yayın saati
kanıtı eksik Tier A'dır. Sabit pilotun olumsuz sonucu bütün hava tahminlerinin
faydasızlığını göstermez. Kod: ml/src/cottonlens_ml/research/weather_exploration.py;
kaynak: ml/src/cottonlens_ml/sources/power_weather.py. Tarihçe:
docs/archive/MASTER_PLAN_20261001.md, 2 Ekim hava pilotu kaydı.

NASA resmi kaynakları 6 Kasım 2026'da incelendi:

| Kaynak | Doğrulanan bilgi | Kabul için açık kalan |
|---|---|---|
| https://gmao.gsfc.nasa.gov/gmao-products/geos-s2s-3/ | v3 Mart 2026'da yayımlanmış mevsimsel tahmin sistemi | Eski tarihlerde v3 çalışmış gibi kullanılamaz |
| https://data.nas.nasa.gov/geos-s2s-3 | 1991–2024, ay başına40ensemble retrospektif altküme kataloğu | Retrospektif yeniden tahmin, o tarihte yayımlanmış gerçek tahmin değildir |
| https://gmao.gsfc.nasa.gov/gmao-products/geos-s2s-3/data-access_geos-s2s-3/ | NRT kullanıcı altkümeleri ve resmi dizin bağlantısı | Ham dosya örneği/ilk yayın/vintage/sürüm sürekliliği kabul edilmedi |
| https://gmao.gsfc.nasa.gov/gmao-products/ | Araştırma, özel sektör ve genel kamu için veri paylaşım beyanı; deneysel ürün, süreklilik garantisi yok | Seçilen dosyanın atıf/lisans/yeniden arşivleme koşulları ayrıca doğrulanmalı |

Resmi bağlantıdan NRT dizinini okuma aracı açamadı. Bu, verinin küresel olarak
yok olduğunu göstermez. Kimseyle iletişim kurulmadı, hesap açılmadı, büyük veri
indirilmedi. Katalog seviyesinde sayısal ürünün varlığına kanıt var; kullanılacak
hücrelerin kabulüne henüz yok.

Sonraki kabul paketi şunları sağlamalı: forecast/hindcast/reanalysis ayrımı,
model sürümü, initialization zamanı, gerçekten erişilebilir olma zamanı, lead,
valid başlangıç/bitiş, ensemble üyesi, değişken/birim/takvim, anomaly referans
iklimi, dosya hash'i, lisans ve revizyon bağı. Initial time tek başına availability
time değildir. Sonradan üretilen1991hindcast'i1991origin girdisi yapmak sızıntıdır.

Coğrafi dönüşüm sonuç görülmeden sabitlenmeli: üretim bölgeleri/poligonları,
alan veya o tarihte bilinen üretim ağırlıkları, ekim/hasat mevsimi, kara-deniz
maskesi, grid eksikleri ve birimler. Aynı takvim ve gecikme kontrolü korunmalı.
Gerekli sayısal örnek ve tarihsel erişilebilirlik kanıtı yokken ekonomik hipotez
ve feature grid'i seçilmedi. Bu araştırma, fiyatın kontrat/PIT açığını kapatmaz.

## İlerideki ürün işi için kabul şartları

Bu bölüm uygulama tasarımı gereksinimidir; backend/frontend değiştirilmedi.

| Görünüm | Zorunlu bilgi | Kabul kontrolü |
|---|---|---|
| Tahmin | provider, kontrat/roll kuralı, fiyat alanı/birim; origin, issued_at, first_observed_at, target, horizon | UNKNOWN ve unavailable açık görünür; boş alan güncelmiş gibi sunulmaz |
| Zaman | UTC saklama ve saat dilimiyle sunum; issue/valid/observed ayrı | Yıl sonlarında takvim yılı/ISO hafta yılı karışmaz; hedef resmi seansa bağlı |
| Kaynak | fiyat tarihi, gerçek alınma saati, gecikme ve son başarılı güncelleme | Eski cache canlı feed etiketiyle gizlenmez; live/backtest/illustrative korunur |
| Revizyon | ilk değer ve sonradan gelen değişiklikler | İlk tahmin yeniden yazılmaz; daha iyi görünen sürüm seçilmez |
| Metrik | formül, signed/absolute ayrımı, birim, örnek sayısı, dönem, ufuk, eksikler, Naive | Aynı kontrat/alan/origin ve gerçekleşme olmadan karşılaştırma üretilmez |
| Aralık | nominal olasılık ve yöntem; gerçekleşen coverage/örnek sayısı | Min/max olasılıklı güven bandı gibi etiketlenmez |
| WASDE/CFTC/on-call | gözlem/as-of, rapor yayını, vintage ve sezon ayrı | Yeni vintage eski origin'e taşınmaz; '<500' sıfır sayılmaz |

## L4 allocator OOM teslimatı

ml/gpu_oom_probe.py yeni süreçte128MiB logical GPU limiti ve256MiB float32
isteğiyle TensorFlow ResourceExhaustedError bekler. Başka hata veya beklenmeyen
başarı kabul edilmez. İkinci temiz süreç aynı GPU'da küçük hesapla recovery
kontrolü yapar. Cihaz açıkça L4 seçilir; CPU fallback yok. Tamamlanmış sonuç
hash'leriyle cache okunur, kısmi/bozuk sonuç kendiliğinden tekrar çalıştırılmaz.

Bu gerçek GPU allocator davranışının sınırlı testi için hazırlanmıştır;
fiziksel VRAM'i tüketme, tüm Colab oturumunu kaybetme, stochastic RNG resume
veya eğitim sonrası model devamını kanıtlamaz. Test henüz Colab'da çalışmadı.
Piyasa/sentetik model fit sayısı0. Yerel testler subprocess kontrol akışını
stub'larla denetler; GPU kanıtı üretmez.

Dosyalar output/l4-oom-handoff-20261106 içinde:
CottonLens_GPU_oom_L4.ipynb, cottonlens-a100-source.zip, source-manifest.json.
Kaynak ZIP adında a100 bulunması mevcut exporter adıdır; notebook açıkçaL4ister.
Yeni L4 Colab runtime aç; notebook'u yükle, L4 seç, hücreleri sırayla çalıştır,
upload hücresinde aynı teslimattaki kaynak ZIP'i seç. Son hücre sonuç ZIP ve
checksum indirir. Bu dosyalar kullanıcı tarafından çalıştırılacaktır. Eski
başarılı restart probe'u yeniden çalıştırma. Kısmi sonuçta aynı namespace'i
silip tekrar deneme; çıktıyı koru ve incelemeye getir.
