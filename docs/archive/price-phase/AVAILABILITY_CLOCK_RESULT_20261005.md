# Availability clock: uygulama ve sonuç — 5 Ekim 2026

**Karar: Naive korunacak; bu sabit Ridge/bilgi düzeninde zamanlama hipotezi için yeni arama açılmayacak.**
Müdahale T+1'de kontrolü küçük ölçüde iyileştiriyor, fakat Naive'yi geçmiyor.
İyileşmenin %87,54'ü, 2020'de geçmiş iç doğrulamanın müdahale ağırlığını sıfıra
indirmesinden geliyor. Ham tahminlerde iyileşme yok. Zamanlama kaybı, bu tarifin
pratik hedefi kaçırmasını açıklayan ana mekanizma olarak desteklenmiyor.

## Dondurulmuş deney

- Profil: `availability-clock-pilot-v1`; deney: `research-availability-clock-pilot-v1`.
- Kaynak kimliği: `1cb25671e4051a42eb004943c439d0c1bd1bfcb3332ca8d2a20121ea8e1cb239`.
- Tasarım kimliği: `5fec0fe3cb15f74a6e477109f08320ff536745332a75e2caeda28aadb2f9ecd9`.
- Market SHA256: `23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c`.
- Referans: `output/full-year/experiments/research-full-year-v1-r2`.
- 2016–2023'te her kol/ufuk için aynı 2.006 origin; 2024+ değerlendirmeye alınmadı.
- Ayrı T+1/T+5 Ridge alpha=1, seed=42, scaled-log hedef, pencere=1;
  3×63 geçmiş iç doğrulama, 21 gözlemde refit, beş gözlemlik etiket olgunlaşması.
- Ağırlıklar yalnız geçmiş doğrulamayla `[0, .25, .5, .75, 1]` içinden seçildi.
- 288 iç + 388 dış = **676 fit**, **32 yıllık çıktı**, **8.024 tahmin satırı**.
  Yeniden `pilot` çağrısı tamamlanmış çıktıları kullanır; yeni fit gerektirmez.
- Kontrolün 24 özelliği, NaN desenleri dahil, referansla mutlak `1e-12` toleransta eşleşti.
  Cotton özellikleri, fiyatlar, hedefler ve origin'ler korundu. Yalnız altı çapraz
  piyasa getirisi ve iki korelasyon müdahale kolunda yeniden hesaplandı.
- Müdahale DXY için 2.005/2.006, WTI için 2.004/2.006 origin'de daha yeni bar
  kullandı. Medyan yaş bir Cotton gözleminden sıfıra indi; iki kolda da bu
  değerlendirme origin'lerinde eskilik sınırı aşımı yok. Müdahale fiilen uygulandı.

Karar anı Cotton kaynak tarihinden sonraki gün 00:15 UTC. Müdahale DXY/WTI barını
kaynak tarihinden sonraki gün 00:00 UTC'de erişilebilir **varsayar**.
Bu, tarihsel Yahoo yayın/teslim/vintage kanıtı değildir. Deney duyarlılık analizidir;
üretim erişilebilirliği veya bağımsız holdout başarısı iddiası taşımaz.

## Ölçülen sonuçlar

Kazanç = `100 × (1 − model fiyat-MAE / aynı-origin Naive fiyat-MAE)`.
MAE birimi cent/lb. Kazanılan yıl sayısında Naive ile eşitlik kazanım değildir.

| Ufuk / kol | Seçilmiş MAE | Naive kazancı | Ham kazanç | Kazanılan yıl | Aktif oran | Ham yön / seçilmiş tüm-origin yön |
|---|---:|---:|---:|---:|---:|---:|
| T+1 kontrol | 1,024306 | −%0,3285 | −%1,8459 | 0/8 | %25,12 | %48,95 / %12,26 |
| T+1 müdahale | 1,022523 | −%0,1538 | −%2,0309 | 0/8 | %12,51 | %50,00 / %6,43 |
| T+5 kontrol | 2,376040 | −%0,8000 | −%4,5101 | 0/8 | %25,12 | %50,50 / %12,56 |
| T+5 müdahale | 2,388760 | −%1,3396 | −%4,9159 | 0/8 | %37,59 | %50,35 / %19,14 |

Naive MAE: T+1 **1,020952**, T+5 **2,357184**. Sıfıra küçültülen tahminler
yön metriğinde sıfır işaretiyle değerlendirilir; düşük seçilmiş yön oranları,
ham yön başarısı veya aktif tahminlerin doğruluğuyla karıştırılmamalıdır.

T+1 müdahalenin kontrole göre ortalama MAE kazancı yaklaşık **%0,1741**.
Yıl sınırlarını koruyan paired bootstrap, 10.000 tekrar, seed=42:

| T+1 karşılaştırması | Blok 20: %95 kazanç aralığı | Blok 60: %95 kazanç aralığı |
|---|---:|---:|
| Seçilmiş müdahale / kontrol | [%0,0147; %0,3374] | [%0,0510; %0,2953] |
| Seçilmiş müdahale / Naive | [−%0,4177; %0,1257] | [−%0,3330; %0,0193] |
| Ham müdahale / ham kontrol | [−%0,5375; %0,1938] | [−%0,5314; %0,1864] |

Her iki blokta Naive kazancı üst sınırı %5'in çok altında. Önceden belirlenmiş
karar tablosunda sonuç **“kontrole katkı var, pratik eşikler sağlanmıyor”** satırına
girer. Bu küçük katkı, başarılı ham tahmin sinyali olarak yorumlanamaz.

T+1'de her iki kol 2016, 2018, 2019, 2021, 2022 ve 2023'te sıfır ağırlık seçti.
2017'de ikisi de .75 seçti ve Naive'ye kaybetti. 2020'de kontrol .75 seçip kaybetti;
müdahale sıfır seçerek Naive'ye döndü. Kontrole karşı toplam hata azalmasının
%87,54'ü bu tek yılın zararından kaçınmadır. Seçim geçmişe dayalıdır; bu gözlem
sonradan ağırlık değiştirme gerekçesi değildir.

En büyük %1 Naive hatası çıkarıldığında T+1 müdahale kazancı −%0,1707,
T+5 müdahale kazancı −%1,4729. Uç gözlemler çıkarılınca pratik üstünlük doğmuyor.
T+5 ikincildir; T+1 başarısızlığını yeniden tanımlamak için kullanılmadı.
Aralıklar sabit tahmin akışına koşulludur; araştırma geçmişindeki bütün insan/model
seçimlerini veya bağımsız yeni dönem belirsizliğini kapsamaz.

## Düzeltilen hatalar ve kanıt sınırları

1. **Legacy LSTM:** artifact `v20260922-1450`, kod `13761b9d30efb6f1827f89246ce9bf6adae45421`.
   Eski kod `test.iloc[59:]` ile LSTM'yi 439 origin'de ölçerken Naive 498 origin'de
   ölçülmüş. T+1 yanlış karşılaştırma +%2,764668; aynı pencere Naive MAE'si
   0,6597719996 ile yeniden kurulan toplu kazanç **−%1,696141**.
   T+5 aynı-pencere toplu kazanç −%4,813858. Tam LSTM tahmin dosyası bulunmadığı
   için tahmin düzeyinde doğrulama ve paired güven aralığı üretilmedi.
   Eski artifact değiştirilmedi. Mevcut sequence motoru yeniden yazılmadı;
   498 origin'in tümünü geçmiş bağlamla koruyan regresyon testi eklendi.
2. **Öğrenme kontrolü:** `research-v2-tf-placement` TCN bilinen-sinyal kazancı
   **−%23,593133** iken genel durum `passed` idi. Yeni politika
   `all-executed-families-v2`, çalıştırılan her ailede ≥%50 şartını uygular;
   eksik/başarısız aile veya eski doğrulanmamış durum geçiş sağlayamaz.
   XGBoost negatif ve küçük örneğe ezberleme kontrolleri korundu. Eski kayıt
   değiştirilmeden ayrı düzeltme raporu üretildi; TCN yeniden eğitilmedi.
3. **Karşılaştırma:** yeni pilot origin, hedef tarihi, ufuk, Cotton fiyatı ve
   gerçekleşen getiri eşitliğini zorunlu tutar. Uyumsuzlukta kesişim alınmaz.
   API ve Model Lab legacy açıklaması bu sınırlamayı açıkça bildirir; üretim
   artifact/API şeması değişmedi.

109 kayıtlık tarihsel envanterde eksik yıllar/kimlikler ve tahmin dosyası varlığı
açıkça işaretlendi. Tamamlanmamış deney negatif sayılmaz. T+5 kaynak testinden
T+1 sonucu çıkarılmaz. Tam sıfır shrinkage ve `[0,0]` aralığı kaynakta sinyal
yokluğu kanıtı değildir. 2016–2023 ve daha önce incelenen 2024+ bağımsız holdout değildir.

## Doğrulama ve teslimatlar

- 49 dar kapsamlı + 83 ilgili ML testi geçti; 29 backend testi geçti; ilgili Ruff kontrolleri temiz.
- Frontend Node 24.19.0 ile `npm ci` ve build geçti. Legacy API yanıtlarını kullanan
  yerel UI testinde 1440/1024/390 px için `scrollWidth − clientWidth = 0`;
  ekran görüntüleri `output/playwright/clock-model-lab-*.png`.
- İlk backend denemesinde sistem Python'unda PyArrow eksikti. Projenin tanımlı
  21.0.0 sürümü ayrı `output/backend-audit-deps` klasöründen kullanılarak testler
  tamamlandı; ML CPU ortamına GPU veya backend paketi eklenmedi.
- Tüm 676 fit kaydının kimliği ve checkpoint checksum'ı doğrulandı. Ham/seçilmiş
  tahminlerden fiyatları bağımsız kuran kontrol, dört kol/ufukta MAE ve Naive'yi
  `1e-12` içinde doğruladı. GPU doğrulaması/eğitimi çalıştırılmadı.
- Başlangıçta kaydedilen **165.781** eski JSON/JSONL/CSV/Parquet dosyasının
  SHA256 karşılaştırmasında değişen veya kayıp dosya yok. Canlı artifact'in
  kendi checksum listesindeki 10 dosya da ayrıca doğrulandı. İlk fitten son
  tamamlanan fite geçen süre 109,74 saniye; testler/prepare/bootstrap bu süreye dahil değil.
- Kaynak kodu gerçek deneyden önce donduruldu. Yeni veri indirilmedi; commit/push,
  canlı model veya eski artifact güncellemesi yapılmadı. Başlangıçtaki çalışma
  ağacı değişiklikleri ayrı snapshot ile korunuyor.

Yerel kanıt yolları:

- Yeni manifest ve kayıtlar: `output/full-year/experiments/research-availability-clock-pilot-v1/`.
- Karar: aynı dizinde `reports/clock-c2c60ae42678de4b.json`.
- Tahminler: aynı dizinde `reports/outer-predictions.csv`.
- Düzeltmeler: `output/availability-clock-implementation/corrections/`.
- Birleşik envanter: `output/availability-clock-implementation/evidence-inventory.json`.
- Bağımsız kontrol: `output/availability-clock-implementation/independent-verification.json`.
- Eski kanıtlar: `old-evidence-checksums.json` ve `old-evidence-verification.json`
  (`output/availability-clock-implementation/` altında; doğrulamanın kapsamı dosyada belirtilir).

Tek sonraki karar: **Naive korunur; bu bilgi/model düzenindeki zamanlama kolu için
yeni grid, kaynak veya GPU araması başlatılmaz.** Bu sonuç bütün olası modellerin
başarısını veya bütün temel kaynaklarda sinyal yokluğunu karara bağlamaz.
