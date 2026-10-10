# T+5: ortak örneklemde eğitim etiketi kontrolü — 10 Ekim 2026

## Önceki sorudan kalan tek ayrım

[PR38](https://github.com/ErayKulkizaga/CottonLensAI/pull/38) tamamlandı.
CT ile öğrenilmiş sabit ham getirinin aynı kontrata taşınması Naive'ı
kurtarmadı: T+5 numeric D0 −%8,664216, blok20/60 üst sınırları negatif.
Bu sonuç, **yalnız ölçüm proxy'sini değiştirme** açıklamasını daraltır;
aynı kontrat etiketiyle öğrenme sınanmış değildir. Yeni model/grid yok.

Bu kontrolün sorusu: **aynı eğitim tarihlerinde, aynı özellik/model ile
CT yerine origin'de sabitlenmiş ikinci kontratın getirisini öğretmek
ham tahminin aynı kontrattaki fiyat-MAE'sini iyileştiriyor mu?**

## Sıfır-fit kurulabilirlik bulgusu — VERIFIED

2021/2022/2023 üç eski dış cohort'a aynı hazır olma kuralı uygulanır.
İç doğrulama, dış seçim kararından önce elde bulunan son189 uygun ve
kaynak açısından olgun T+5 etiketi:3×63. Her gerçek takvim refit'inde
en az500 ortak olgun eğitim etiketi gerekir. En erken eğitim sayıları
sırasıyla32/279/528: yalnız2023 desteklenir. Yıl seçimi dış fiyat/hata
değerlerini kullanmaz. Bu **yeni iç cohort protokolüdür**, eski eksiksiz
3×63 bloklarının değişmeden aktarımı değildir.

2023 dış cohort değişmez:246 makbuz,242 eşleşmiş hedef,4 açık eksik/
uygunsuz kayıt. Haziran21/26 origin fiyatının varsayımsal saati geçtir;
Kasım23 origin fiyatı yoktur; Kasım16'nın T+5 hedef fiyatı yoktur.
Eksikler doldurulmaz veya sessiz ortak kesişimle silinmez. Bu dört
kaydın gelecekteki sonuçları yıl/seçim kuralını değiştirmez.

İç origin aralığı2022-03-23–2022-12-22. Üç63'lü blok sırasıyla3/4/3
refit kovası içerir. Dış12 kova ile kol başına22, toplam **44 piyasa
fit'i**, ayrıca en fazla **1 ayrı bilinen-sinyal sentetik kontrol**.
21 kova takvimdeki Cotton sıra numarasıyla belirlenir;63 uygun kaydı
63 kesintisiz seans gibi sıkıştırmayız. Kovada ilk uygun gün geç olsa
da eğitim kesimi asıl21-seans sınırıdır.

İlk özel readiness taslağında filtrelenmiş CSV sıra etiketleriyle
history index'inin hizalanması hatalıydı; sıfır uygun yıl üretti.
Taslak ve çıktısı korunur. Konum/date eşitliği denetimli R2 ve ayrı
araştırma modülü44/528 sonucunda eşleşti; regresyon testi eklendi.
Bu taslak hiçbir fit veya piyasa kararı üretmedi. Eksik son hedef
tarihleri de NaT eşitliğiyle karşılaştırılır; toleransla tarih kabulü yok.

## Kilitlenecek protokol

Profil `named-label-control-t5-v1`; iki kol `ct` ve `named`.

- Veri: PR35 frozen history, PR37 ikinci contemporaneous kontrat
  etiketleri ve doğrulanmış kaynak/teslimat kanıtları. Yeni veri yok.
- Hedef: Cotton takvimindeT+5; kontrat adı origin'de seçilir ve hedefe
  kadar değişmez. Geleceğin ön vade adı kullanılmaz.
- Özellik: aynı28 numeric D0 alanı; Cotton ve curve temsili değişmez.
- Model: Ridge alpha1,seed42,window1, aynı train-only medyan/imputation/
  standardizasyon, global train-standardized log-return, expanding.
- Ortak train: özgün120 warmup; origin<cutoff,target_date5<cutoff,
  uygun pozitif fiyatlar ve hedef quote'un varsayımsal erişimi refit
  kararından geç değil. İki kolda train tarihleri/özellikleri aynıdır;
  yalnız `target_return_5` değiştirilir. Ham CT price alanı eğitimde
  sabit tutulur; native training-price MAE named kontrat performansı
  olarak yorumlanmaz.
- Seçim: kol başına geçmiş3×63 üzerinde eşit blok ağırlıklı göreli
  named fiyat-MAE; shrinkage[0;0,25;0,5;0,75;1], eşitlikte küçük ağırlık.
  Ham ağırlık1 birincildir; seçilmiş sonuç ikincildir.
- Karar saati: Cotton kaynak tarihi+1 gün00:15UTC. AMS için önceki
  deneyin `max(reference,displayed publication day)+1 gün00:00UTC`
  erişim ve Final vintage varsayımı aynen sürer. **Tarihsel UTC/ilk
  sürüm kanıtı değildir.** Kaynak kabulü0; üretim/live değişmez.
- Fiyat: initial_named×exp(predicted_log_return). Origin fiyatı karar
  saatinde yok/geç ise fiyat tahmini yayımlanamaz; log çıktı/makbuz
  korunur. Hedef eksikse skor bilinmez. Naive aynı initial_named fiyat.
- Çalışma: mevcut Experiment/Ledger; ayrı source/data/environment
  identity ve namespace, tekCPU süreç/iki thread/30 dakika oturum.
  Hesaplanmış ama kopyası yarım fit yeniden eğitilmez. Tavan aşılmaz.

## Önceden belirlenen karşılaştırma

Birincil: ham CT mutlak hatası−ham named mutlak hatası; pozitif named
lehine. Aynı242 named hedefte MAE, Naive'a kazanç ve yön ayrıca verilir.
Seçilmiş shrinkage, aktif oran, düz tahminler ayrı kalır. Blok20/60,
10.000 tekrar,seed42; bütün246 sıra korunur, bilinmeyenlerin örneklem
ağırlığı0, bölen her tekrardaki gerçek uygun kayıt sayısıdır. Eksik
hedefler sıfır hata sayılmaz. En büyük%1 Naive hatası çıkarılan duyarlılık
ayrı gösterilir; birincil sonucu değiştirmez.

İki blok aralığının altı pozitif ve named ham kazanç≥%5 ise yalnız
**bu geçmiş yılda** eğitim hedefi mekanizmasını destekleyen aday bulgu.
Named'ın Naive kazanç üst sınırı her iki blokta%5 altında kalırsa
bu sabit örneklem/bilgi/model düzeninde etiket düzeltmesi pratik hedefi
kurtarmıyor. Diğer durumda belirsiz. Bu sonuç bütün modellerin veya
bilgi kaynaklarının sinyalsizliğini kanıtlamaz. Aralıklar yıllarca
yapılan insan/model seçimlerini kapsamaz; etiket eksikliğinin seçilim
yanlılığı da çözülmüş sayılmaz.

**Tek görülmüş yıl**, sekiz yıllık6/8 kapısını ölçemez. %5 ve T+5%55
yön eşikleri değişmez; bağımsız holdout, ekonomik işlem veya üretim
başarısı iddiası yok. Sonuçtan sonra otomatik ek grid/ufuk/yıl açılmaz.

## Teslimat durumu

17 dar regresyon testi ve Ruff başarılı. Geniş ML doğrulamasında
1.048 test geçti,3 skip; iki mevcut NASS concat FutureWarning var.
GPU/backend/UI değişmedi; yerel GPU/backend/UI build çalıştırılmadı. Kaynak/girdi/ortam kimliği, checksum-bound ön kayıt ve GitHub'dan
taze geri okuma tamamlanmadan gerçek fit başlamaz. Henüz yeni fit0;
bu belge eğitim sonucu değildir. Eski plan/veri/tahmin/ledger korunur.
