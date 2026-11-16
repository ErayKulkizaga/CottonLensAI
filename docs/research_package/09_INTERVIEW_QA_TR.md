# 09 · Görüşme soruları ve cevapları (Türkçe)

Cevaplardaki her sayı bu paketin diğer dosyalarında kaynağıyla birlikte yer alıyor.

## Genel

**Proje ne?**
ABD pamuk piyasası üzerine iki aşamalı bir araştırma.
- Birinci aşamada (Codex ile) ICE pamuk vadeli fiyatını 1 ve 5 işlem günü sonrası için tahmin etmeye çalıştım.
- İkinci aşamada (Claude ile) ABD ve Teksas pamuk üretimini, USDA'nın aylık tahmininden daha isabetli tahmin etmeye çalıştım.

**En önemli sonucun ne?**
USDA tahminlerini temkinli revize ediyor: tahmini bir ay düşürdüyse, nihai rakam çoğunlukla biraz daha düşük çıkıyor. Bu gecikmeyi düzelten kural önceden sabitlendi, sadece geçmiş yıllarla eğitildi ve hiçbir ayarı değiştirilmedi. 2002–2025 arasında USDA'nın her aylık raporunda (1 Eylül, Ekim, Kasım, Aralık) Teksas, ABD ve eyalet bazında ortalama hatası USDA'nınkinden düşük çıktı: 12 ölçümün 12'si. 2002–2011'i hiç görmeden test ettim.

**Bu "USDA'yı her zaman geçtin" demek mi?**
Hayır. Fark %3–11 arası, her yıl kazanmıyor ve 8 hücrenin yalnızca 3'ünde istatistiksel olarak anlamlı. Aynı kural mısır ve soyada ön kayıtlı testte başarısız oldu (T46). Örneğin Aralık'ta ABD'de 21 yılın 14'ünde daha yakın, 7'sinde daha uzak; Eylül'de ABD'de 9'a 10 kaybediyor. Önceden yazdığım resmi testler de teknik olarak geçmedi: T44 Eylül'ü de şart koşuyordu, T45'te Teksas Aralık'ta 3–3 berabere kaldı.

**Kaç model eğittin?**
- Birinci aşamanın kayıt defterinde 32.934 tamamlanmış eğitim var: 28.245'i gerçek piyasa verisiyle, 4.597'si sentetik kontrol, 92'si denetim ve tekrar üretim.
- Ayrıca `research-market-v1` ablasyonunda 20.360 alt-eğitim (12,2 saat hesaplama) var; bu sayı deftere ayrıca yazılmamış.
- İkinci aşamada 120 GPU CNN eğitimi ve 1.000'i aşkın yıllık yeniden eğitim (bu sonuncusu yaklaşık) var.

**Kaç deney?**
- 133 fiyat/risk deneyi, 569 tarif kaydı, 41 araştırma profili.
- Üretim tarafında 45 numaralı deneme (yaklaşık 60 varyant).
- Toplam 178 deney.

**+%5 hedefini ve %53 / %55 yön eşiklerini kim koydu?**
- Ben Codex'ten zor, kolay ulaşılamayacak bir hedef koymasını istedim. Sayısal eşikleri Codex belirledi; 21 Eylül'de koda girdi (`2eb44a0`).
- Bunlar pratik kabul kuralları, istatistiksel anlamlılık testi değil. Sayıların tek tek nasıl türetildiği kayıtlarda yok; ayrıca bootstrap güven aralığı ve 6/8 yıl tutarlılığı da şart koşuldu.

**Uygulamayı gösterebilir misin?**
- Evet, yerelde. FastAPI backend, Angular arayüz (Dashboard, Model Lab, Replay, Sensitivity, Why), PostgreSQL, hepsi Docker Compose ile açılıyor.
- Docker Desktop açıkken repo kökünde `docker compose up --build`, sonra `http://localhost:8080`.
- Bir yere deploy edilmedi. Uygulamadaki ana fiyat tahmini Naive; deneysel XGBoost sadece duyarlılık sayfasında.

**Link 1 olay çalışması hangi aşamaya ait?**
- 15 Kasım'da Claude ile yapıldı; iki aşama arasındaki köprü. Codex döneminin bir parçası değil.

**Bu fark istatistiksel olarak anlamlı mı?**
- Kısmen. Yıl bazında permütasyon testinde 8 hücrenin 3'ü tek yönlü p < 0,05 (Ekim ABD 0,02, Kasım Teksas 0,03, Aralık ABD 0,02). Diğerleri 0,08–0,38 arası; bu örneklem büyüklüğünde şanstan ayırt edilemiyor.
- Çoklu karşılaştırma düzeltmesi yapmadım; yapılırsa daha da zayıflar.

**Aynı kural başka ürünlerde çalışıyor mu?**
- Hayır. Bunu ön kayıtla test ettim (T46). Kuralın tek bir ayarını değiştirmeden USDA'nın mısır ve soya tahminlerine uyguladım, 2002–2025.
- Sonuç: her iki üründe 3 ayın yalnızca 1'inde USDA'dan iyi; havuzlanmış kazanç +0,03 puan, p = 0,28. Başarısız.
- Yorum: mısır ve soyada USDA zaten %1–3,6 hata yapıyor ve revizyonlarında bu kuralın kullanabileceği bir momentum yok. Pamuktaki etki pamuğa özgü (Teksas'ta büyük terk edilen alan, geç hasat) ya da kısmen şans. "USDA'nın genel bir davranışını bulduk" diyemem.
- Testi yaparken iki veri hatası çıktı (bir "Other States" satırı, bir raporda yapışık iki sütun). İkisini de veri tarafında düzelttim, ön kayda tarihli not olarak ekledim ve bozuk çalıştırmaları da sakladım.

**Bu fikir yeni mi?**
- Hayır. USDA'nın pamuk tahminlerinde "smoothing" (revizyonların aynı yönde devam etmesi) yayımlanmış bir bulgu: Isengildina-Massa, MacDonald & Xie (2012, JARE). 2013 tarihli bir çalışma bu etkiye göre düzeltme yapmanın sadece bazen işe yaradığını ve katsayıların istikrarsız olduğunu gösterdi. Goyal & Adjemian (2023, AJAE) ise bunun nedeninin bilinçli temkinlilik değil, verinin pahalı ya da gürültülü olması olabileceğini söylüyor.
- Benim katkım fikir değil test: bu bulguyu sızıntı kontrollü, ön kayıtlı, kör yıllar içeren bir protokolle bağımsız olarak yeniden sınadım. Sonuç literatürle uyumlu: 2002–2025'te küçük bir kazanç var ama zamanla küçülüyor, çoklu karşılaştırma düzeltmesinden sonra hücre bazında anlamlı değil ve mısır/soyada yok.

**Yayımlanmış düzeltme yönteminden iyi mi?**
- Genel olarak hayır. Aynı yıllarda, sabit terimli ve 10 yıllık ya da genişleyen pencereli bir düzeltmeyle karşılaştırdım. Benim kuralım 8 hücrenin yalnız 2'sinde (Eylül ve Ekim ABD) en iyi; diğerlerinde yayımlanmış tarzdaki yöntem daha iyi.

**Sonuç tek bir yıla mı dayanıyor?**
- Hayır. Her yılı tek tek çıkardığımda 8 hücrenin yalnız 1'inde (Kasım ABD) işaret değişiyor. Ama kuralın katsayısı 2000'lerin başından 2025'e doğru küçülüyor; Eylül'de negatife dönüyor. Etki zayıflıyor.

**Bu işleri kim yaptı?**
- Hedefleri ve kısıtları ben koydum; mühür açma, push etme, bulutta iş başlatma gibi geri alınamaz her adımı ben onayladım. Kodu ve deneyleri Codex (1. aşama) ve Claude (2. aşama) yazıp çalıştırdı. Her iddiayı dosyalara karşı kontrol ettirdim, hataları kayıtta düzelttirdim. Ayrıntı: `14_DECISIONS_AND_COMPUTE.md`.

**En önemli bulgun ne? (güncel cevap)**
- USDA'nın Teksas'taki hatası 2012'den sonra karakter değiştirdi. Önceden hata verim kaynaklıydı. Artık hasat edilen alandan geliyor: hasat alanı hatası yaklaşık %3,7'den yaklaşık %15'e çıktı.
- USDA, sezon içinde terk edilecek alanı sabitlenmiş bir şekilde tahmin ediyor. Anket tarihinde herkesin görebildiği "zayıf + çok zayıf" ürün oranına, nihai terkin ancak üçte ikisi kadar tepki veriyor.
- Sadece geçmiş sezonlarla eğitilen tek değişkenli bir model, 2008'den beri USDA'nın Teksas alan hatasını her raporda yaklaşık %40 azaltırdı.
- Aynı imza Oklahoma–Kansas pamuğunda da var. Yön aynı ve 4 ayın 4'ünde daha iyi; ama ön kayıttaki %5 eşiğini kıl payı kaçırdı (p = 0,056), o yüzden "doğrulandı" demiyorum. Kışlık buğdayda yok; yani bu pamuğa özgü bir terk olgusu.
- 2026 Teksas hasat alanını kilitledim: 3.258.029 akre, USDA'nın Kasım rakamından %16,5 düşük. Mayıs 2027'de belli olacak.

**Momentum kuralı eski yıllarda tuttu mu?**
- Kısmen. 2000–2025 verisiyle donmuş katsayılar 1985–1999'da tutmadı. Ama kural sadece o dönemin kendi geçmişiyle yeniden eğitildiğinde, hiç görülmemiş 1987–1999'da 8 hücrenin 7'sinde USDA'dan iyi. Aralık ABD'de 10 sezonun 9'unda daha yakın. Havuzlanmış p = 0,089, anlamlı değil.
- Yorumum: etki tekrar ediyor ama büyüklüğü döneme göre değişiyor. Bu, 2013 makalesindeki "katsayılar istikrarsız" bulgusuyla aynı.

## Model ayrıntıları

**14.723 XGBoost eğitiminde ne denendi?**
- **7.320'si** Colab T4 GPU'daki büyük aramadan (round A): her ufuk için 128 aday, max_depth 1–8, eta 0,005–0,15, min_child_weight 1–100, alpha 0–9,3, lambda 0,001–95, subsample ve colsample 0,6–1,0. Hedef olarak ölçekli log getiri ve fiyat farkı, kayıp olarak kare ve mutlak hata; 24 özellik; T+1 ve T+5. İç bloklarda fiyat MAE'si ile seçim, en iyi 5 aday 3 farklı seed ile doğrulandı.
- **6.729'u** dokuz veri kaynağı keşfinden: FX 1.020, CFTC 1.008, hava 996, FAS 996, NASS 984, ürün fiyatları 972, WASDE 528, on-call 141, AMS 84. Standart keşiflerde aynı sabit "derinlik 2" tarifi kullanıldı ki yalnızca bilgi değişsin. Her kaynak 0, +1 ve +5 seans gecikmeyle denendi; bilgi geç gelse de işe yarıyor mu diye.
- **Kalan 668'i:** tam yıl çalışması (360), sentetik zayıf sinyal kontrolü (132), kayıp fonksiyonu karşılaştırması (MSE ve MAE, 54) ve aynı kontrat deneyleri.
- **Arama nasıl yapıldı:** adaylar NumPy ile deterministik rastgele çekildi. En iyi 5 aday 17 ve 101 seed'leriyle yeniden eğitildi (42 zaten vardı). Kazanan 3 seed'in ortalamasıyla seçildi; dış tahmin de 3 seed'in ortalaması.
- **En iyi sonuç** +%0,76 (T+5, yön %57,5, 8 yılın 5'i); hedef +%5 idi. Hiçbiri geçmedi.

**LSTM bir kez mi denendi?**
Hayır. Kayıtlarda en az 42 LSTM eğitimi var (22 Eylül'ün sayısı hariç):
- 22 Eylül: 6 aday. Bu çalışmanın tahmin dosyası kayıp (DG-008). Eski "+%2,76" iddiası farklı pencereleri karşılaştırıyordu, geçersiz sayıldı. Aynı pencerede −%1,70 (T+1) çıkıyor.
- 23 Eylül v1: 20 eğitim. 4 aday × (4 dış dönem + 1 son doğrulama).
- 24 Eylül düzeltilmiş v2: 21 eğitim.
- Defterde 1 sentetik kontrol: boru hattı bilinen bir sinyali bulabiliyor mu diye. Genişlik 32, öğrenme oranı 3e-3, batch 64, pencere 20.

v1 mimarisi:
- Girdi: 60 gözlem × 24 özellik.
- Tek LSTM katmanı (32 veya 64 hücre) → Dropout (0,10 veya 0,20) → Dense(32, ReLU) → Dense(2). T+1 ve T+5 getirisi aynı ağdan çıkıyor.
- Eğitim: Adam lr 0,001, batch 64, en çok 100 epoch, standartlaştırılmış hedeflerde MAE kaybı, early stopping (patience 10, en iyi ağırlıklara dönüş), seed 42.

Sonuç: T+1 −%1,09, T+5 −%0,01. Son doğrulama bölmesinde en iyi epoch 1 oldu: doğrulama kaybı 0,922'den 11. epoch'ta 0,979'a çıktı. Diğer dönemlerde en iyi epoch farklı (örneğin 2).
- MLP, LSTM ve TCN için 48'er adaylık bir tur (pencere 20/60/120) tasarlandı ama çalıştırılmadı, çünkü ağaç modelleri bile iç doğrulamadaki kazancı dış yıllara taşıyamamıştı.

**CNN'i neden denedin, neden olmadı?**
- You ve arkadaşlarının 2017'deki yöntemine dayanıyor. Her ilçe-yıl bir "görüntü" oluyor: 6 MODIS bandı × 22 sekiz günlük dönem × 32 aralıklık histogram.
- Mimari: 3 konvolüsyon bloğu (32/64/128 kanal, BatchNorm, ReLU, dropout 0,25), global ortalama havuzlama, 64 nöronluk katman.
- Eğitim: Adam (lr 1e-3, weight decay 1e-4), batch 64, en fazla 60 epoch, önceki yılda erken durdurma (patience 10), 5 seed. Kaggle GPU'da toplam 120 eğitim.
- Sonuç: 1 Ekim'de Teksas MAPE 18,4 (USDA 11,0). Seed'ler arasında 4 puan fark çıktı; 2.753 etiketli örnek bir CNN için az. Karışım modelinde ağırlığı sıfır oldu. Bir sonraki adım mısır/soya verisiyle transfer learning olurdu.

**Gradient boosting ayarların neydi?**
scikit-learn HistGradientBoostingRegressor: max_iter 300, learning_rate 0,04, max_leaf_nodes 15, min_samples_leaf 30, l2_regularization 2,0. Hedef, ilçenin dönüm başı üretiminin kendi geçmiş ortalamasından logaritmik sapması. Özellikler, ilçenin kendi geçmişine göre NDVI anomalileri.

**Volatilite tarafında ne yaptın?**
- Bir sonraki günün kare getirisini tahmin ettim. EWMA (λ = 0,94) referans, kayıp fonksiyonu QLIKE.
- Modeller: HAR (3.048 eğitim), GARCH(1,1) Student-t (707), EWMA ölçeği.
- "Own-level" HAR kolu geçmiş verilerde QLIKE'ı +0,0177 iyileştirdi; iki bootstrap güven aralığı da pozitif, 8 yılın 7'si pozitif. Bu kol ön kayıtta ikincil koldu; birincil DXY kolu sonuçsuz kaldı.
- Bunun için günlük otomatik bir ileri test kuruldu. Test iki kez yayın yaptı (31 Ekim ve 4 Kasım 2026), sonra zamanlanmış görevler kapatıldı; ileri test sonucu yok. Dürüst ifade: "geçmiş testlerde geçti, ileri testi kurduk ama sürdürmedik".

## Yöntem ve dürüstlük

**Veri sızıntısını nasıl önledin?**
Her verinin ne zaman yayımlandığını kodda kural olarak yazdım (MODIS kompozit bitişi + 8 gün, Landsat çekimi + 16 gün, hava kesim tarihinden 3 gün önce vb.). Bu kurallar gerçek sorunlar yakaladı:
- yıl içinde kesinleşmiş ekili alanın kullanılması;
- USDA raporundan sonra kaydedilmiş FSA dosyaları;
- yayın tarihi olmayan sigorta verisi (kaynak tamamen elendi);
- yanlış kontrat ayı taşıyan AMS dosyaları.

**Neden K-fold kullanmadın?**
Zaman serisinde rastgele bölme geleceği eğitime sızdırır. Her yıl yalnızca kendinden önceki yıllarla eğitildi (rolling origin). Fiyat tarafında iç içe doğrulama, etiket olgunlaşması ve purge da vardı.

**Mühürlü test ne demek?**
2022–2024 ve 2025'i en sona sakladım. Testin kuralını yazıp GitHub'a push ettikten sonra bu yılları birer kez açtım. İki test de başarısız oldu ve sonuçları olduğu gibi kaydettim. 2000–2011'deki USDA tahminlerini ise hiç çekmemiştim; momentum kuralını orada kör test ettim.

**İlk "başarı" neden bozuldu?**
USDA 2014–2021'de çoğunlukla fazla tahmin yapıyordu ve geçmiş sapmayı düzelten model işe yarıyordu. 2022, 2024 ve 2025'te USDA düşük tahmin yaptı ve aynı düzeltme hatayı büyüttü. Bu, sabit bir sapma terimi kullanmanın aslında rejimin süreceğine yapılmış bir bahis olduğunu gösterdi. Momentum kuralı bu yüzden sabit terim içermiyor.

**Uydu verisi işe yaramadı mı?**
Kısmen.
- Landsat'ın başarısız tarla payı, Teksas'ta gerçekten terk edilen alanı r ≈ 0,8 ile yakaladı.
- Ama USDA'nın %40 yanıldığı 2023'te, sonradan terk edilen tarlalar 1 Eylül ve 1 Ekim'de hâlâ yeşildi.
- İlçe bazında NDVI anomalisiyle kurulan model, USDA'yı kullanmayan modeller içinde en iyisiydi: 1 Ekim'de Teksas'ta 10,3 / USDA 11,0. ABD toplamında ise geride kaldı.

**Kaç uydu görüntüsü işledin?**
- 28.977 Landsat sahnesi (43.485 sahne tarandı), pamuk tarlası noktalarında 806,5 milyon piksel gözlemi.
- 2.278 MODIS kompoziti; 6 bantla 13.668 bant görüntüsü.

**Bulutta ne kullandın?**
- Kaggle CPU ve GPU: 20'den fazla iş, komut satırından otomatik çalıştırıldı.
- Her iş birimi (bölge-yıl, katman-seed) ayrı dosya olarak tutuluyor; kesinti olursa bitmiş birimler yeniden hesaplanmıyor.
- Fiyat aşamasında Colab T4 GPU kullanıldı.

**Fiyat neden tahmin edilemedi?**
- Günlük pamuk getirisi bu ufuklarda rastgele yürüyüşe çok yakın. 113 deneyin medyanı −%0,65, en iyisi +%0,22.
- Sentetik kontroller boru hattının sinyali bulabildiğini gösterdiği için sorun koddan kaynaklanmıyor.
- Olay çalışması da bunu destekledi: USDA rapor günlerinde fiyat 1,42 kat daha çok hareket ediyor (p = 0,004), ama USDA'nın revizyonu hareketin yönünü tahmin etmiyor.

**2026 için ne yaptın?**
16 Kasım 2026'da tahmini GitHub'a kilitledim (commit `18c3bc7`): ABD 12.763.510 balya; USDA Kasım tahmini 12.794.000. Aralık düzeltmesinin formülü de donduruldu. Mayıs 2027'de nihai rakamla karşılaştırılacak.

**Tekrar yapsan neyi farklı yapardın?**
- Önce benchmark'ın davranışını (USDA revizyonlarını) modellerdim.
- Daha az ama önceden kilitlenmiş deneme yapardım.
- Derin öğrenmeye ancak veri yeterli olduğunda geçerdim.
- Her verinin yayın zamanını ilk günden kaynağıyla kanıtlardım.
