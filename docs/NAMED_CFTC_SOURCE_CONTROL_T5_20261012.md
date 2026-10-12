# Named T+5: korumalı CFTC içeriği kontrolü

## Soru ve kapsam

Sabit XGBoost/named-kontrat hedefinde CFTC'nin üç sayısal pozisyon alanı,
aynı kaynak yaşı ve eksiklik bilgisine ek tahmin katkısı sağlıyor mu?
[Önceki denetim](NAMED_CFTC_INFORMATION_AUDIT_20261012.md) eski CFTC
deneyinin farklı CT=F hedefini kullandığını doğruladı. O 1.855 fit yeniden
çalıştırılmaz. Tamamlanan 132-fit sentetik kontrol de tekrar edilmez.

Bu çalışma tek, daha önce incelenmiş 2023 kohortuna aittir. Bağımsız holdout,
sekiz yıl kapısı, üretim modeli veya kaynak kabulü değildir. Gerçek tarihsel
UTC yayını ve ilk vintage kanıtlanmadı. Korumalı saat altında içerik katkısını
sınar; daha erken erişim, bütün kaynak temsilleri veya bütün modeller hakkında
karar vermez. Naive'ı geçmek bir test veya süreç tamamlanmasıyla sağlanmış sayılmaz.

## Dondurulacak tarif

- Profil `named-cftc-source-control-t5-v1`; deney
  `research-named-cftc-source-control-t5-v1`.
- Private CFTC denetim Release'inden taze geri doğrulanmış gerçek named
  geçmişi ve mevcut `positions-2010-2023.csv`. Yeni veri indirme yok.
- Aynı 246 origin, 242 uygun hedef, dört bilinmeyen hedef korunur.
  Eğitim günleri etiket olgunluğu kuralından yeniden üretilir; sıkıştırılmış
  takvim veya eksik özellik nedeniyle origin düşürme yok.
- Aynı 32 alan + `cftc_age_L1`, `cftc_missing_L1`,
  `cftc_managed_net_share_L1`, `cftc_producer_net_share_L1`,
  `cftc_managed_share_change_L1`: iki kolda da 37 alan.
- Numeric kol özgün değerleri taşır. Mask kol yalnız son üç alanın sonlu
  değerlerini sıfırlar. NaN, yaş, missing, diğer bütün sütunlar aynıdır.
- CFTC report-date +7 takvim günü, sonraki Cotton gözlemi, L1 ve mevcut
  10-gözlem carry sınırı aynıdır. Legacy +3-gün alanları açılmaz; L2/L6 yok.
  Bilinen gecikme/revizyon maskeleri değişmez. Saat varsayım olarak kalır.
- Ayrı bir öğrenme motoru yok: mevcut `Experiment.fit`, ledger, preprocessing,
  standardized log-return, named T+5 hedefi, olgunlaşma ve refit düzeni.
- XGBoost depth2, eta.03, min_child_weight20, alpha0, lambda1, seed42,
  100 ağaç; window1, expanding history, cadence21. Hiperparametre araması yok.
- Aynı 3×63 geçmiş doğrulama; `[0,.25,.5,.75,1]` ağırlığı yalnız geçmişte,
  eşit blok-relative fiyat-MAE ortalaması ve en küçük ağırlıkla eşitlik kırma.
- 44 piyasa + dört güçlü sentetik mekanik kontrol; toplam en fazla 48 fit.
  Bir CPU süreç, en fazla iki thread, 30 dakika oturum. Hatalı denemeler de
  bütçeye girer; duraklamada mevcut checkpoint korunur, otomatik tarif değişmez.

## Eğitim kapısı ve doğrulama

Kod/testler bittikten sonra yeni source/data/environment/design kimliği
dondurulur. Eski modeller/data/evidence değişmez. Seçili girdiler private
GitHub denetim paketinin checksum envanterine bağlanır; gerçek named geçmişi
ile candidate projeksiyon tekrar hesaplanır. Sentetik öğretici hedefleri
piyasa deneyine aktarılmaz.

Ön kayıt, sabit raporlayıcı ve bağımsız doğrulayıcı GitHub'a teslim edilir;
taze indirme/restore makbuzu olmadan `prepare` ve `pilot` çalışmaz. Kaynak,
ortam, veri, tasarım veya yayın makbuzu değişirse cache reddedilir.
Her modelin scaler/target ve native tahmini geçmiş train satırlarıyla yeniden
doğrulanır. Yarım çıktı tamamlanmış deney olarak raporlanmaz.

57 ilgili regresyon testi ve Ruff geçti. Testler sentetik/veri bütünlüğü
kapsamındadır; piyasa becerisi kanıtı değildir. Backend/UI/artefact API şeması
değişmedi; GPU ve gerçek piyasa eğitimi henüz çalışmadı.

## Önceden belirlenen analiz ve karar

Birincil: **ham mask-MAE eksi numeric-MAE**, pozitif değer kaynak katkısıdır.
Ham ve küçültülmüş Naive kazancı, yön doğruluğu, aktif oranı ayrı verilir.
En büyük %1 Naive hatası çıkarılmış duyarlılık ikincildir. Her karşılaştırma
aynı origin, target-date, kontrat, fiyat, hedef ve uygunluk sırasını zorunlu
tutar; sessiz kesişim alınmaz.

Paired moving-block bootstrap: 10.000 tekrar, seed42, blok20 ve60. Dört
bilinmeyen hedef kaybı sıfır-bookkeeping weight ile tam 246 takvim slotunda
kalır; gerçek sıfır kayıp gibi ortalamaya girmez. Bu aralıklar geçmişteki
model/insan seçiminin çoklu karşılaştırma yükünü kapsamaz.

1. İki ham kaynak-katkısı aralığının alt sınırı pozitif, numeric ham Naive
   kazancı ≥%5 ve yön ≥%55: yalnız araştırma adayı; 6/8 ve ileri doğrulama
   sağlanmış sayılmaz.
2. Katkı aralıkları pozitif fakat pratik nokta eşikleri sağlanmıyor: katkı var,
   hedef çözülmedi; otomatik grid/kaynak ekleme yok.
3. Pozitif katkı gösterilmedi ve numeric ham Naive kazancının iki üst sınırı
   <%5: bu sabit saat/temsil/tarif mevcut pratik hedefi kurtarmıyor.
4. Diğer sonuçlar belirsizdir; kaynakta sinyal yokluğu sonucu çıkarılmaz.

Sonuç ilk karar tablosuna yerleştirilmeden yeni model/grid açılmaz. Üretim
modeli, live kayıtları, Drive özgünleri veya PR merge bu teslimata dahil değildir.
