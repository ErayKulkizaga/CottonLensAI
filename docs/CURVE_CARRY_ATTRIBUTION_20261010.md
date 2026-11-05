# Eğri katkısı hangi fiyat bileşeninde? — sıfır-fit karşıolgu

**VERIFIED:** Tamamlanan [T+5 deneyi](CONTRACT_CURVE_T5_RESULT_20261010.md)
ile aynı 749 origin ve aynı tahminler kullanıldı. Kaynak çiftli 743 origin,
altı bilinmeyen korunur; 74 geçiş origin'i 15 olaya aittir. Yeni fit0;
model, seçim, özgün hedef, gate ve birincil karar değişmedi. Protokol ve
analiz kodu hesap öncesi checksum ile donduruldu. Bu yeni tahmin deneyi değildir.

## Tam olarak ne hesaplandı?

Origin fiyatı `c`, özgün log hedef `r`, gelecekte ilk sırada görünen kontratın
origin fiyatıyla mevcut ilk kontrat arasındaki log fark `g` olsun.
Özgün hedef fiyat `y=c*exp(r)`; karşıolgusal fiyat `y0=c*exp(r-g)`.
Her iki kolun kayıtlı tahmin fiyatı **sabit** tutulur.

`Delta = |y-f_mask| - |y-f_numeric|`

`Delta0 = |y0-f_mask| - |y0-f_numeric|`

`DeltaCarry = Delta - Delta0`

Pozitif değer numeric kolunun kontrol hata toplamını azalttığını gösterir.
Bu, kontrat farkı hedefte bulunmadığında aynı iki tahmin arasındaki hata
farkının nasıl değiştiğini ölçer. Mutlak hata doğrusal değildir; ayrı
bileşen MAE'lerini toplamak veya bunu ekonomik PnL ayrıştırması saymak yanlıştır.
Float32 kaynak fiyat yuvarlaması ayrıca doğrulanır, veri “düzeltilmez”.
`g` geleceğin kontrat kimliğini kullanır: **oracle tanı, origin feature'ı değil**.
Karşıolguya göre eğitim/refit yapılmadı; yeni hedefte OOS beceri ölçülmedi.

## Geçişlerdeki pozitif toplam kontrat farkı çıkarılınca tersine döndü

| 74 geçiş origin'i — kontrol eksi numeric hata toplamı | Özgün proxy | Gap çıkarılmış karşıolgu | Karşıolgusal fark |
|---|---:|---:|---:|
| Ham D0 | +24,657265 | −3,367436 | +28,024702 |
| Seçilmiş D0 | +4,829044 | −3,836961 | +8,666005 |
| Ham D1 | +13,592705 | −1,958986 | +15,551690 |
| Seçilmiş D1 | +3,338605 | −2,590032 | +5,928637 |

669 aynı-kontrat origin'inde `g=0`: bütün katkılar **birebir aynı kaldı**.
Ham D0 burada zaten −94,456540 hata katkısı üretiyordu; ham D1 −80,984343.
Bu nedenle roll, bütün-origin başarısızlığın ana açıklaması değil. Altı
bilinmeyene gap uydurulmadı: ham D0 özgün katkısı −7,346250, karşıolgu null.
Kaynak çiftli 743 origin'de ham D0 toplam −69,799275'ten −97,823976'ya,
D1 −67,391638'den −82,943328'e gidiyor. Bunlar tanısal toplamlar;
tam 749 origin'in kayıtlı MAE/kararı aynen duruyor.

## Gözle görülebilen mekanizma: Ekim 2022

Kaynak tablosu 3 Ekim 2022'de Oct-22 `92,14`, Dec-22 `84,20`; 10 Ekim'de
ilk kontrat Dec-22 `88,23` gösteriyor. Sürekli hedef **92,14 → 88,23**
düşüşü ölçerken, iki tabloda bulunan aynı Aralık kontratı **84,20 → 88,23**
yükseliyor. Kaynak belge SHA256'ları:
`8ed4ea5eff97dc69708c23853eac78247ade81e5d843f40ad248b50850b842bc` ve
`8a8286cc45a3d223809052cc9d675da5c54e2e8aaec54f78d9b53bd18485492a`.

3 Ekim origin'inde ham mask tahmini log +%3,0277, numeric −%0,0406.
Numeric kol proxy'de kontrol hatasını `2,869789` azaltıyor; gap çıkarılınca
aynı tahmin farkı hatayı `2,869789` artırıyor. Bu örnekte Aralık kontratı
origin'de de tabloda var; yine de analizdeki genel kontrat seçimi geleceğin
ilk kontratını kullandığından bir işlem kuralı olarak sunulmaz.

10 Ekim 2022'de gözlenen bu tek değişime temas eden beş origin, ham D0'ın
geçiş katkısının **%64,3543**'ünü, seçilmiş D0'ın **%81,3046**'sını oluşturuyor.
Örnek sonuçlara baktıktan sonra açıklama için seçildi; yeni cohort/threshold
veya bağımsız pozitif deney değildir. Bütün 15 olayın sonuçları arşivdedir.

## Kendi açıklamamızı çürütmeye çalışınca ne kalıyor?

**STRONGLY SUPPORTED:** Bu sabit eğri tarifinin geçişlerdeki pozitif
**toplam** katkısı, proxy'nin kontrat farkına duyarlı. Küçük proxy katkısını
aynı-kontrat fiyat yönü/pozisyon becerisi gibi sunmak yanlış olur. Ufku
uzatmak ya da küçültmeyi kaldırmak bu tarifin genel hedefini kurtarmadı.

**Karşı kanıt:** Ham D0'ın 15 olayından 10'unda özgün katkı pozitif; karşıolguda
hâlâ sekiz olay pozitif. D1'de altıdan yediye çıkıyor. Dolayısıyla “eğride
bütün sinyal yalnız roll'dur” sonucu **NOT SUPPORTED**. D0 toplamındaki
işaret dönüşümü özellikle Ekim 2022 olayından etkileniyor; 15 bağımlı olay
güvenilir genel kaynak üstünlüğü kanıtı değil. Ayrı güven aralığı verilmedi.

**Alternatif açıklama:** Proxy hedefinde eğrinin meşru bir bilgisi olabilir;
bu hedef için eğitilmiş sabit tahminleri başka hedefte puanlamak doğal olarak
o bilgiyi cezalandırır. Bu nedenle karşıolgu, yeniden eğitilmiş aynı-kontrat
modelinin becerisini veya tüm kaynakta bilgi yokluğunu ayıramaz. Mevcut
669 aynı-kontrat origin'indeki olumsuz katkı ise bu hedef değişimine bağlı
değildir. Kalan soru genel “daha büyük model” değil, **origin'de tanımlı
kontratın hareketine dair kullanılabilir bilgi gerçekten ölçülüyor mu?**

**Karar:** Sabit gerçek-eğri T+1/T+5 tarifini büyütme; Naive korunur.
Kaynak performansı **INCONCLUSIVE**, karşıolgusal mekanizma **VERIFIED**.
Saat/ilk vintage varsayımlı; geçmiş yıllar bağımsız holdout değil. Model
yayını, pozitif net getiri, hedef başarısı veya 6/8 gate kabulü yok.

## Doğrulama ve koruma

Bağımsız scalar/55-haneli Decimal programı 2.996 satırdaki özgün hata ve
743 kaynak çifti × dört karşılaştırma = 2.972 ayrıştırma/karşıolguyu yeniden
hesapladı. 669 aynı-kontrat origin'i × dört = 2.676 kimlik kontrolü,
altı bilinmeyen × dört = 24 satır, 15 olay × dört = 60 olay özeti ve
üç yıl × dört = 12 yıllık özet doğrulandı; bunlar bağımsız gözlem sayıları
değildir. Yedi ret kontrolü: hedef tarihi, oracle feature işareti,
karşıolgu değeri, eksik origin, tahmin girdisi, quote girdisi ve eski çıktı
üzerine yazma. Ruff geçti. İlk doğrulayıcı/kontrol sürümlerinin yalnız lint
düzeltmesi gereken kopyaları ve başarılı makbuzları ayrıca korunur; bilimsel
analiz kodu veya sonuç değişmedi, tolerans gevşetilmedi.

Önceki 162 çalışma/518 tarif/78 kanıt dosyası, 1.000 ham girdi ve T+5 sonucu
korunur. ML source `20d0c1869afc35b9c83b34cfd3fc8de8ec90a56ebd64c1cf3dcb3b085685d7b3`
değişmedi; bu tanı eğitim motoruna/profile'a eklenmedi. Çalışan model/canlı
görev etkilenmedi. Sicile sıfır-fit tanı eklenir; trials nesneleri değişmez.

## Sonraki tek bilimsel adım

**Yeni eğitimden önce origin'de bilinen, ufuk boyunca sabit kontrat hedefinin
mevcut veriyle kurulabilirliğini sıfır-fit denetle.** Aynı 749 origin'de
önceden tanımlı ilk ve ikinci kontrat için, T+1/T+5'te aynı isimli quote
bulunuyor mu; orijinal Cotton seansları, eksiklikler, fiyat alanı ve erişim
varsayımı neyi kanıtlıyor? Sonuçlara göre en iyi kontratı seçme; yalnız
iki sabit tanımın kapsama/kimlik matrisi. Kaybolan eski kontrat quote'una
yeni kontrat fiyatı koyma, forward-fill veya otomatik origin düşürme yapma.

Bu PR #33'ün **gelecekte ilk olacak kontrat** ayrıştırmasından farklıdır:
kontrat kimliği origin'de sabitlenir. Bir label uygulanabilirliği denetimi,
yeni model başarısı deneyi veya eski hedeflerin yeniden yazılması değildir.
`history.py check --horizon 5 --query contract` eski hedef denetimi, ön kayıt
ve tamamlanmış T+5'i gösterdi; bunlar tekrar eğitilmeyecek. Yeni fit/grid,
veri indirme veya varsayımlı kaynağın tarihsel kabulü kendiliğinden açılmaz.

## Ayrı kanıt paketi

[Sayısal kanıt](../research/evidence/curve-carry-attribution-20261010.json),
[manifest](../research/evidence/curve-carry-attribution-release-20261010.json)
ve [Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/curve-carry-attribution-20261010).
29 üye /2.230.347 byte; ZIP SHA256
`227ce52023dba9b1f91bbe0724fdf13df30c99024559bbe15bd4917a929260a1`.
Arşivdeki RESULT.md bu teslimat ekinden önce dondurulan bilimsel rapordur.
Yeni sicil toplamı 163 çalışma/518 tarif; eski nesneler aynen korunur.

GitHub'dan yeniden indirilen 29 üyenin checksum'ı, rapor/satırların exact
replay'i, bağımsız Decimal makbuzu ve yedi ret kontrolü tekrar doğrulandı.
Yerel ve fresh makbuzlar byte olarak aynı; fit0.
[Teslimat kanıtı](../research/evidence/curve-carry-attribution-delivery-20261010.json).
Bu makbuzun gözlemlediği başlıkta 12 CI kontrolünden dokuzu başarılı,
üçü çalışıyordu; sonraki teslimat commit'inin CI'sı ayrı değerlendirilir.
Obsidian kanonik proje notuna sonuç ve karşıolgu yorum sınırı eklendi.
