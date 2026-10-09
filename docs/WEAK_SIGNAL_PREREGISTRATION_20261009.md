# Zayıf sinyal kalibrasyonu — sıfır-fit ön kayıt

Profil `weak-signal-control-v1`, deney `research-weak-signal-control-v1`.
Bu çalışma sentetiktir; piyasa becerisi, gerçek tarihsel veri erişimi veya
bağımsız holdout doğrulaması değildir. Texas/NASS negatif piyasa sonucu değişmez.

## Soru ve sabit tasarım

Mevcut Ridge ve geçmişe dayalı shrinkage, kendi girdilerinde doğrusal olarak
temsil edilebilen küçük bir etkiden ne kadarını koruyor? Eski gürültüsüz,
yaklaşık %99,75 kazanımlı öğrenme kontrolleri bu soruyu sınamadı.

- Referans: checksum bağlı tamamlanmış `research-nass-regional-t1-pilot-v1`.
- Aynı `numeric_D0` 37 özellik, eksiklikler, Cotton fiyatları ve 2.006 origin;
  2016–2023, T+1. 2024+ kullanılmaz. Kaynak saati/vintage varsayımsal kalır.
- Ridge alpha=1, model seed=42, pencere=1, standardized log-return hedefi.
  Expanding history, ortak warmup, beş gözlemlik maturity kuralı; 3×63 geçmiş
  doğrulama, 21 gözlemde refit ve ağırlıklar `[0,.25,.5,.75,1]` aynı kalır.
- 10 gürültü seed'i: 1201–1210. Her seed'de `null` ve `injected` koşulları.
  İki koşulda aynı gürültü; girdiler aynı. Model yalnız geçmiş sentetik
  etiketlerden öğrenir; `oracle_return_1` model girdisi değildir.

Sentetik log-getiri `r_t = beta*z_t + sigma_t*epsilon_t`; null koşulunda beta=0.
`z_t`, Texas–ulusal good/excellent farkının erken dönem medyan/std ile
ölçeklenmiş halidir; kaynak bilinmiyorsa 0. `sigma_t`, mevcut geçmişe dayalı
20 gözlemlik Cotton oynaklığıdır; [.005,.05] aralığına kesilir. Geçersiz
oynaklıkta yalnız erken dönem medyanı kullanılır. Gürültü bağımsız standart
normaldir; tüm tarihlerde seed sırası dondurulur.

Beta, yalnız 2010–2014 ve ortak warmup sonrası 1.139 satırdan analitik olarak
hesaplanır: fiyat-MAE oracle kazanımı bu kalibrasyon döneminde %5 olacak.
553 satırda kaynak biliniyor. Beta yaklaşık .00845356935; 2015+ hedeflerine
bakarak ayarlanmaz. Değerlendirme dağılımındaki beklenen/gerçekleşen oracle
katkısı ayrıca raporlanır; %5'e ulaşması zorunlu değildir.

Oracle tahmini beta*z'dir: lognormal fiyat dağılımının koşullu medyanı.
Gerçek fiyat `C_t*exp(r_t)`, Naive `C_t`, model `C_t*exp(predicted_return)`.
Bu etiketler tutarlı bir işlem gören fiyat yolu oluşturmaz; bağımsız
regresyon/öğrenme tanısıdır. Gerçek piyasa etiket dosyası değiştirilmez.

## Önceden kilitlenen yorum

Her koşul/seed için ham, seçilmiş ve oracle MAE, yıllık sonuç, aktif/yön
oranları, en büyük %1 Naive hatası çıkarılmış duyarlılık raporlanır.
Yıl sınırlarını koruyan paired bootstrap: 10.000 tekrar, seed42, blok20/60.
Bu aralıklar geçmiş bütün araştırma seçimlerini kapsamaz. On seed üzerinden
kalibre edilmiş güç veya evrensel model kapasitesi iddiası yapılmaz.

Karar sırası:

1. Enjekte koşulunun ortalama gerçekleşen oracle kazanımı <%3 veya bir seed'de
   oracle kazanımı <=0: `INVALID_CALIBRATION`; model başarısızlığı çıkarılmaz.
2. En az iki null seed'de seçilmiş kazanç >=%5, yön >=%53, >=6/8 yıl ve iki
   blokta pozitif alt sınır: `NULL_CONTROL_FAILS`; düzen yeniden denetlenir.
3. Ham ve seçilmiş sonuçların her biri >=8/10 seed'de pozitif ve aynı-seed
   oracle kazanımının medyan olarak >=yarısını koruyor:
   `RAW_AND_SELECTED_RECOVER`.
4. Yalnız ham sonuç bu koşulu sağlıyor: `RAW_ONLY_RECOVERS`; seçim/shrinkage
   bilgi kaybı adayıdır.
5. Ham ve seçilmiş sonuçlar koşulu sağlayamıyor:
   `FIXED_RECIPE_DOES_NOT_RECOVER`; negatif piyasa sonuçları kaynakta sinyal
   yokluğunu ayırt ettirmez.
6. Diğer örüntü: `INCONCLUSIVE`. Yeni fit otomatik açılmaz.

Başarılı kalibrasyon, yalnız bu doğrusal, sabit ve kodlanmış etkiyi yakalama
becerisini gösterir; farklı temsilleri, rejimleri veya tüm kaynakları elemez.
Normal gürültü burada koşullu ortalama log-getiri ile fiyat-MAE için koşullu
medyanı uyumlu kılar. Başarı, gerçek getirilerdeki asimetri veya hedef/loss
uyumsuzluğunu elemez; bu deneyin model için elverişli varsayımıdır.
Piyasa %5 MAE / %53-%55 yön / 6/8 yıl eşikleri değiştirilmez; Naive korunur.

## Çalıştırma ve teslimat

Ön kayıt/kod/test kimliği GitHub'da yayımlanmadan fit başlamaz. 20 senaryo ×
(72 inner + 97 outer) = **3.380 sentetik fit**, 160 yıllık çıktı ve 40.120 OOS
satırı. Yeni piyasa fit'i 0. Bir süreç, en fazla iki thread; doğrulama dahil
oturum başına 30 dakika. Kimlik değişirse cache reddedilir; yarım çalışma aynı
kimlikte checkpoint'ten sürer. Başarısız girişimler bütçede sayılır.

Mevcut CPU ortamında `ml/full_year_cpu.py` ile sırasıyla
`prepare → pilot-plan → pilot → compare → report`; profil yukarıdaki,
`--reference-root` tamamlanmış Texas deney dizini. Açık `--drive-root` ve
`--cpu-environment` verilir. `pilot` checkpoint korur; `report` ayrıca fit
yapmadan katsayı/scaler/etiket maturity/seçim/fiyat-MAE/aralık replay'i yapar.

Kaynak snapshot'ı, senaryo geçmişleri, manifest, tüm fit payload'ları,
ham/seçilmiş/oracle tahminleri ve bağımsız kontrol checksum bağlı Release'te
saklanır. Sicil piyasa ve sentetik kanıtı ayırır. Sonuca göre tek sonraki karar
verilir; gece ileri kayıt görevi bu çalışmanın parçası değildir.
