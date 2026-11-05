# Gerçek vade farkı T+1 sonucu — pratik hedef karşılanmadı

[Ön kayıt](CONTRACT_CURVE_PROTOCOL_20261010.md), PR #32. Tek süreç/iki thread,
252 piyasa +1 ayrı sentetik Ridge fit'i; 12 çıktı, 749 ortak origin/kol,
2.996 tahmin satırı. Arama/yeni veri/GPU yok. Naive ve canlı görev korunur.
Source `84d40007…`, registration `e2fac4fa…`; kod commit'i `8c89cc1b3a045ae069fe1539768474e3dc2efe95`.

**VERIFIED:** 12/12 ağırlık geçmiş üç iç blokta 0 seçildi. Dört kolun seçilmiş
fiyat-MAE'si 1,4978238094: Naive ile aynı, aktif tahmin %0. Tüm-origin yön
%0,2670 yalnız iki sıfır gerçekleşmeden gelir; aktif yön tanımsızdır. Bu sayı
“model %0,27 biliyor” veya “Naive aşağı tahmin ediyor” şeklinde yorumlanmaz.

## Ham tahminler küçültmenin arkasında saklanmadı

| Kol | Ham fiyat-MAE | Naive kazancı | Ham yön | Naive'ye karşı yıllar |
|---|---:|---:|---:|---:|
| mask D0 | 1,5183097655 | −%1,367715 | %50,0668 | 0/3 |
| numeric D0 | 1,5384385802 | −%2,711585 | %50,3338 | 0/3 |
| mask D1 | 1,5166723945 | −%1,258398 | %50,3338 | 1/3 |
| numeric D1 | 1,5320151683 | −%2,282736 | %49,1322 | 0/3 |

Ham numeric D0'ın yıllık Naive kazancı 2021/2022/2023'te
−%2,585003/−%2,800559/−%2,677116; D1'de
−%1,719315/−%2,778749/−%1,919108. Ham aktif oran bütün kollarda %100.

| Karşılaştırma | Nokta kazancı | Blok20 %95 aralık | Blok60 %95 aralık |
|---|---:|---:|---:|
| Ham D0 numeric–mask | −%1,325738 | [−%3,023366; +%0,578924] | [−%2,729506; +%0,184420] |
| Ham D1 numeric–mask | −%1,011608 | [−%2,037738; +%0,350930] | [−%1,929245; +%0,065182] |
| Ham numeric D0–Naive | −%2,711585 | [−%4,838498; −%0,498497] | [−%4,524886; −%0,702921] |
| Ham numeric D1–Naive | −%2,282736 | [−%4,134912; −%0,339407] | [−%3,852635; −%0,579536] |

En büyük sekiz Naive hatası çıkarılınca ham numeric D0/D1 kazancı
−%3,212651/−%2,615817; birkaç büyük gözlem gizli üstünlüğü bastırmıyor.
Ham numeric–mask yıllık katkısı her iki gecikmede üç yılın tamamında negatif.

**Karar:** `FIXED_RECIPE_BELOW_PRACTICAL_GOAL_SOURCE_INCONCLUSIVE`.
Birincil seçilmiş kaynak katkısı **INCONCLUSIVE**: iki kol sıfıra döndüğü için
paired `[0,0]` bilgi yokluğunu veya sıfır etki için yüksek gücü kanıtlamaz.
Ham sonuç, **bu tek doğrusal log-oran/çekirdek/Ridge tarifinin** pratik T+1
hedefini kurtarmadığını destekliyor. Küçültmeyi kaldırmak çözüm değildir:
ham eğri tahminleri Naive'den daha kötü. Eğride bütün modellerce kullanılabilir
bilgi bulunmadığı veya T+1'in doğası gereği imkânsız olduğu sınanmadı.
Üç yıl 6/8 gate'i değerlendirmez; gate değişmedi, otomatik yayın yok.

## Vade değişimleri gerçekten bu başarısızlığı açıklıyor mu?

**VERIFIED, sonuç-sonrası betimleyici tanı:** aynı 749 origin'in 743'ünde
iki referans gününün tabloları var. 743/743 origin ve hedef Close, tablodaki
ilk kontrat fiyatıyla 1e−5 toleransta eşleşiyor. 15 origin'de ilk vade adı
değişiyor; altı origin'in kontrat çifti bilinmiyor. 728 aynı-vade gözleminde
quote oranı ile hedef log-return azami farkı 9,76e−8 (mevcut fiyat yuvarlaması).

Basılı ilk-vade değişiminden ±5 Cotton gözlemi uzakta **584 origin** kalıyor.
Burada ham numeric D0/D1 Naive kazancı −%2,475314/−%1,916725. Dolayısıyla
gözlenen OOS başarısızlığı sadece 15 geçiş origin'ine yüklenemez. Bu,
2010–2020 eğitim label'larının tamamen temiz olduğu veya tüm roll etkilerinin
silindiği ispatı değildir. Teslim edilen kontratın ertesi gün fiyatı tabloda
kaybolabildiği için geçişte aynı-kontrat getirisi uydurulmadı.
Gelecek kontrat adları yalnız bu betimleyici tanıda kullanıldı; feature,
kohort, seçim, hedef veya birincil karar değiştirilmedi. Bu yeni performans
deneyi ya da doğrulanmış ICE roll takvimi değildir.

## Bağımsız doğrulama ve korunan kanıt

253 kayıtlı model çıkarımı yeniden fit olmadan **birebir** tekrarlandı;
252 piyasa preprocessing/target durumu ve olgun train cohort'u doğrulandı.
12 shrinkage seçimi 108 iç fit'in kayıtlı tahminlerinden yeniden hesaplandı.
16 D0/D1 ham/seçilmiş, kontrol/Naive, blok20/60 aralığı ayrı resampling
uygulamasıyla doğrulandı; bütün kol fiyat/yön/aktif metrikleri bağımsız
fiyat dönüşümünden hesaplandı. Rapor ve CSV exact replay geçti.

Çıkarım doğrulayıcısı pencere1'in native C-contiguous matris düzenini ve
metriklerin float64 fiyat dönüşümünü korur. İlk doğrulayıcının F-order/float32
price hesaplaması küçük yuvarlama farkıyla reddedildi; model/tahmin/rapor
değiştirilmedi, tolerans gevşetilmedi. Sentetik bilinen-sinyal kontrolünde
MAE iyileşmesi %99,7669; bu güçlü sentetik öğrenme kanıtıdır, piyasa becerisi
veya zayıf eğri etkisini çıkarabildiği kanıtı değildir.

Eski 1.000 ham kaynak checksum'ı, sicil/fit tarifi/kanıt kayıtları korunur.
Saat ve ilk vintage hâlâ varsayım; 2021–2023 araştırma geçmişidir. Aralıklar
önceki insan seçimlerini kapsayan bağımsız holdout olarak sunulmaz.
Sonuç paketi ön kayıttan ayrı, ek kayıt olarak saklanır.

[Sayısal sonuç](../research/evidence/contract-curve-result-20261010.json) ve
[ayrı sonuç Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/contract-curve-result-20261010):
2.834 üye /15.432.088 byte; SHA256
`197de751ff52afb4f0e86a66d9943ee7f51f5a213e786619796f71c784f5deba`.
Kaynak/input/ortam, 253 native payload, 12 seçim/çıktı ve 2.996 tahmin
korunur. Ön kayıt Release'i değiştirilmez. Sicile yalnız tamamlanmış bu
deney ve dört piyasa/ayrı bir sentetik tarif eklenir; eski158 çalışma/508
tarif aynen kalır. Toplam159 çalışma/513 tarif; sınıf INCONCLUSIVE.

GitHub'dan yeniden indirilen 2.834 güvenli üyenin checksum'ları ve **253
çıkarım/12 seçim/16 aralık/exact rapor+CSV** tekrar doğrulandı; yerel ve fresh
doğrulama makbuzları byte olarak aynı, yeni fit0.
[Teslimat makbuzu](../research/evidence/contract-curve-delivery-20261010.json).
Kod commit'inin sekiz CI kontrolü başarılı; sonuç/delivery commit'lerinin
devam eden kontrolleri bu makbuzda geçmiş gibi gösterilmez.

## Sonraki tek bilimsel adım

**Bu sabit T+1 tarifini büyütmeyi durdur.** Sonraki iş, aynı gerçek eğri,
kontrol ve erişim varsayımıyla **yalnız ufku T+5'e değiştiren ön kayıt
incelemesi**: bilgi frekansı/ufuk uyumsuzluğu ile bu temsilde kullanışlı bilgi
yokluğunu ayıracak mı? Eski spot-basis T+5 buna cevap vermedi. Yeni fit
kendiliğinden başlamaz; önce aynı 749 label/cohort, purge, overlap, güç ve
bütçe doğrulanır. Pozitif T+5 ancak bu tarifte ufka bağlı katkıyı destekler;
negatif T+5 tüm eğride bilgi yokluğu veya daha karmaşık model ihtiyacını
tek başına karara bağlamaz. Yeni kaynak/grid önerisi yok.
