# Gerçek vade farkı T+5 sonucu — ufku uzatmak hedefi kurtarmadı

[Dondurulmuş ön kayıt](CONTRACT_CURVE_T5_PROTOCOL_20261010.md), PR #34.
`research-contract-curve-t5-pilot-v1`; source
`20d0c1869afc35b9c83b34cfd3fc8de8ec90a56ebd64c1cf3dcb3b085685d7b3`,
registration `aeed8a21350c322a34125f080a9c5adc77029ce4407e1d3e8356a2a2774ca8b6`.
Kod commit'i `c5322b2b3eeea83a578add22d9fbf42abaa082af`.
252 piyasa +1 ayrı T+5 sentetik fit; tek süreç/iki thread, 12 çıktı,
2021–2023'te 749 aynı origin/kol, 2.996 tahmin satırı. Model, özellikler,
veri ve origin'ler T+1 ile aynı; yalnız ufuk değişti. Arama ve yeni veri yok.

## Sonuç ve sınır

**VERIFIED:** Naive fiyat-MAE `3,4351135172`. Geçmiş iç doğrulama bütün
kollarda 2021/2022/2023 için `[0; 0,25; 0]` seçti. T+1 ağırlıkları
aktarılmadı. 498 origin sıfır tahmin, aktif oran %33,511348. Ana numeric D0
seçilmiş MAE `3,4539812231`: Naive kazancı **−%0,549260**, yıl kazanımı 0/3.
Aktif yön %46,215139; tüm-origin yön %15,620828. Tüm-origin düşük yönü
yalnız aktif tahmin başarısıyla karıştırmayın; Naive “aşağı” tahmini değildir.

| Kol | Seçilmiş MAE | Naive kazancı | Ham MAE | Ham Naive kazancı | Ham yön |
|---|---:|---:|---:|---:|---:|
| mask D0 | 3,4517720561 | −%0,484949 | 3,5809062610 | −%4,244190 | %47,797063 |
| numeric D0 | 3,4539812231 | −%0,549260 | 3,6839042919 | −%7,242578 | %49,132176 |
| mask D1 | 3,4516926894 | −%0,482638 | 3,5801700397 | −%4,222758 | %47,797063 |
| numeric D1 | 3,4582044833 | −%0,672204 | 3,6771724326 | −%7,046606 | %49,265688 |

Ham bütün kollarda aktif oran %100, Naive'ye karşı kazanılan yıl 0/3.
Ham numeric D0 yıllık Naive kazancı −%2,696667/−%8,392204/−%10,174777;
D1 −%2,977624/−%8,702348/−%8,133446. En büyük sekiz Naive hatası
çıkarılınca ham D0/D1 kazancı −%8,410464/−%8,047585; büyük gözlemler
çıkarılınca gizli üstünlük ortaya çıkmıyor.

| Paired karşılaştırma | Nokta kazancı | Blok20 %95 aralık | Blok60 %95 aralık |
|---|---:|---:|---:|
| Seçilmiş D0 numeric–mask | −%0,064001 | [−%0,776748; +%0,794353] | [−%0,722776; +%0,664106] |
| Ham D0 numeric–mask | −%2,876312 | [−%6,888297; +%1,900537] | [−%6,848943; +%1,789119] |
| Seçilmiş numeric D0–Naive | −%0,549260 | [−%1,449594; +%0,536663] | [−%1,416794; +%0,469089] |
| Ham numeric D0–Naive | −%7,242578 | [−%12,862282; −%0,727987] | [−%13,134426; −%0,406356] |

Aralıklar yılları koruyan paired moving-block, 10.000 tekrar, seed42;
fark aralığı merkezlenmiş basic yöntemle hesaplanır. Eski insan seçimlerini
kapsayan bağımsız holdout değildir. Küçük kaynak etkisi hâlâ belirsiz;
bu sabit seçilmiş tarifte %5 pratik Naive üstünlüğü desteklenmiyor.

**Karar:** `FIXED_RECIPE_BELOW_PRACTICAL_GOAL_SOURCE_INCONCLUSIVE`.
Kaynak katkısının sınıfı **INCONCLUSIVE**; bu, bu tarifin pratik hedefi
karşılamadığı bulgusunu geri almaz. Saat/ilk vintage varsayımlı. Üç yıl
6/8 yıl gate'ini ölçemez; %5 MAE/%55 T+5 yön eşikleri değiştirilmedi.
Üretim/pozisyon becerisi ve tarihsel leakage yokluğu iddiası yok; Naive korunur.

## Katkı hangi tür origin'de ortaya çıkıyor?

**VERIFIED, yalnız sonuç-sonrası betimleyici:** 74 origin'de origin–T+5
basılı ilk kontrat adı değişiyor, 669'unda aynı, altısında quote çifti yok.
Bu 74 origin yalnız 15 geçiş olayını içeriyor; 74 bağımsız olay değildir.
Gelecekteki kontrat adı hiçbir feature, seçim veya origin filtresine girmedi.

Ham D0 numeric–mask toplam mutlak-hata katkısı:

| Sonradan belirlenen grup | Origin | Kontrol MAE | Numeric MAE | Toplam hata azalması |
|---|---:|---:|---:|---:|
| Kontrat değişimi | 74 | 3,9869820370 | 3,6537757504 | +24,657265 |
| Aynı kontrat | 669 | 3,5300438868 | 3,6712345293 | −94,456540 |
| Bilinmeyen | 6 | 4,2437930863 | 5,4681681634 | −7,346250 |
| Tümü | 749 | 3,5809062610 | 3,6839042919 | −77,145525 |

D1'de aynı işaret: geçiş +13,592705, aynı kontrat −80,984343,
bilinmeyen −5,263154, toplam −72,654792. Alt gruplara güven aralığı veya
bağımsız pozitif deney sınıfı atanmadı; birincil bütün-origin karar değişmedi.

**STRONGLY SUPPORTED:** Sadece günlük ufku beş gözleme uzatmak bu eğri
temsilini kullanışlı genel fiyat tahminine dönüştürmedi. Hem T+1 hem T+5'te
ham kol daha kötü; küçültmeyi kaldırmak çözüm değil. T+5'te kazanç işareti
kontrat değişimlerinde yoğunlaşıyor, genel harekette kayboluyor.
**PLAUSIBLE:** Eğrinin küçük katkısı sürekli proxy hedefindeki kontrat
değişimi bileşenini kısmen yakalıyor; pozisyon değerindeki hareketi değil.
Hedef ayrıştırması bu mekanizmayı mümkün kılıyor, fakat mevcut grup tablosu
nedenselliği veya kaynakta tüm kullanılabilir bilginin yokluğunu kanıtlamıyor.
**NOT SUPPORTED:** Daha büyük model, daha fazla veri/compute, küçültmeyi
kaldırma veya roll'u bütün başarısızlığın ana nedeni sayma.

## Kanıtların doğrulanması

Sonuç doğrulaması ayrı, fit girişleri kapalı programla yapılır. Kaynak saati
7.040 seçimde, özgün Cotton T+5 hedefi 749 origin'de; 253 native çıkarım,
12 geçmiş shrinkage kararı, 16 paired aralık, bütün fiyat/yön/aktif metrikleri,
rapor/CSV ve ayrı geçiş açıklaması yeniden hesaplanır. Geçiş sayıları ve hata
toplamları ayrıca bağımsız quote/CSV hesabıyla doğrulanır. Doğrulamada yeni fit0.
Güçlü sentetik kontrol %99,766921 kazanç: piyasa becerisi veya küçük eğri
etkisini çıkarma gücü değildir. Yerel ML 1.031 passed/3 skipped, Ruff;
GPU çalıştırılmadı. Kaynak PR #34'ün son başlığında sekiz CI başarılı.

Eski 1.000 ham girdi, 161 çalışma, 513 tarif ve 75 kanıt dosyası korunur.
Sıfır-fit ön kayıt Release'i değiştirilmez. Ayrı sonuç kanıtı ve GitHub
teslimat doğrulaması [sicile](../research/README.md) ek kayıt olarak girer.

## Sonraki tek bilimsel adım

**Yeni fit yerine mevcut tahminlerde sıfır-fit proxy/carry ayrıştırması.**
Aynı 749 origin ve saklanan ham/seçilmiş tahminleri koru. Quote çifti olan
743 origin için önceki denetimdeki cebirle hedefi origin vade farkı ve
gelecekteki aynı kontratın hareketi olarak ayır; iki kola da aynı fiyat
dönüşümüyle yalnız betimleyici hata hesabı uygula. Altı bilinmeyeni ayrı
tut, tam cohort sonucunu silme. Gelecekteki kontrat adı sadece oracle tanıda;
yeni target eğitimi, seçim, roll takvimi veya tahmin başarısı sonucu üretme.

Soru: 74 geçiş origin'inde gözlenen katkı, proxy'nin kontrat farkı
bileşeni çıkarıldığında sürüyor mu? Kaybolursa görünen katkı fiyat yönü
kanıtı sayılamaz; sürerse aynı-kontrat bileşeninde hedefli bilgi sorusu açık
kalır, yeni eğitim otomatik açılmaz. Bu PR #33'ün **gap-only oracle/Naive**
hesabını tekrarlamaz: ilk kez tamamlanmış T+5 **aday–kontrol tahminlerinin**
bileşenlere duyarlılığını ölçer. `history.py check --horizon 5 --query contract`
ilgili eski hedef denetimi/ön kaydı buldu; eşleşme yokluğu yenilik ispatı
sayılmadı. Yeni grid/model/veri veya canlı yayın bu karardan doğmaz.

## Ayrı sonuç arşivi

[Sayısal kanıt](../research/evidence/contract-curve-t5-result-20261010.json),
[checksum manifesti](../research/evidence/contract-curve-t5-result-release-20261010.json)
ve [sonuç Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/contract-curve-t5-result-20261010).
2.842 üye /16.987.796 byte; ZIP SHA256
`0f8c4a78946e115d847564d2a5f806147a0b1dbfd55978b4ef367824fca87919`.
Arşiv içindeki RESULT.md bilimsel raporun teslimat ekinden önce dondurulan
sürümüdür. Sicile yalnız bu tamamlanmış deney ve dört piyasa/ayrı bir
sentetik tarif eklendi: toplam 162 çalışma/518 tarif; önceki nesneler aynen kaldı.
