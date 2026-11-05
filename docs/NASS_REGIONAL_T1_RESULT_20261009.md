# Texas/NASS T+1: tamamlanan sınırlı deney

**VERIFIED:** 676 piyasa fit'i + ayrı 1 sentetik Ridge kontrolü tamamlandı.
2016–2023'te dört kolun her birinde aynı **2.006 origin**, toplam **8.024 tahmin**
ve 32 yıllık çıktı var. Karar **FIXED_RECIPE_BELOW_PRACTICAL_GOAL**:
Naive korunur; bu tarifi büyüten grid veya otomatik ek eğitim açılmaz.
[Sonuç/kimlik](../research/evidence/nass-regional-result-20261009.json),
[checksum manifesti](../research/evidence/nass-regional-result-release-20261009.json),
[model ve tahmin paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/nass-regional-result-20261009).

## Hangi soruya cevap veriyor?

Değişmeyen [ön kayıt](NASS_REGIONAL_T1_PREREGISTRATION_20261009.md) yürütüldü.
Sabit Ridge alpha1, pencere1, expanding geçmiş, 21 gözlemde refit;
geçmiş 3×63 blokta yalnız shrinkage `[0,.25,.5,.75,1]` seçilir.
24 Cotton/çapraz-piyasa özelliği, rapor yaşı, ulusal kondisyon ve otomatik
eksiklik göstergeleri iki kolda aynıdır. Sayısal kol yalnız sekiz Texas
kondisyon/gelişim değerini ekler; mask kolunda aynı hücreler biliniyorsa0,
bilinmiyorsa null kalır. Hiçbir origin eksiklik nedeniyle atılmadı.

D0: basılı rapor tarihi+1 gün00:00 UTC varsayımsal erişim; karar Cotton
kaynak tarihi+1 gün00:15 UTC. D1 bir ek kayıtlı Cotton kararı bekler.
İkisinde aynı D0'ya bağlı10 gözlemlik eskilik sınırı var; kışa taşıma ve yeni
raporda olmayan aşamayı eskisinden doldurma yok. Tarihsel ilk vintage/erişim
**doğrulanmadı**. Sonuç varsayıma bağlı duyarlılık kanıtıdır; gerçek PIT kabulü,
bağımsız holdout veya üretim becerisi değildir. 2024+ deneye girmedi.

## Sonuçlar

MAE fiyat biriminde; pozitif kazanç daha iyidir. Ortak Naive MAE **1,020952242**.

| Kol | Seçilmiş MAE | Naive kazancı | Ham Naive kazancı | Seçilmiş yıl kazanımı | Aktif oran / aktif yön |
|---|---:|---:|---:|---:|---:|
| mask_D0 |1,022054000|−%0,1079|−%3,4374|2/8|%37,29 / %50,27|
| numeric_D0 |1,021232431|−%0,0274|−%3,7557|1/8|%37,39 / %48,93|
| mask_D1 |1,022518911|−%0,1535|−%3,5698|2/8|%49,90 / %49,65|
| numeric_D1 |1,023561679|−%0,2556|−%3,9653|2/8|%49,90 / %49,55|

Tüm-origin seçilmiş yön doğruluğu numeric D0/D1 **%18,54/%24,93**;
sıfır tahminler dahil olduğu için aktif yönle karıştırılmaz. Ham sayısal yön
**%49,85/%50,05**, her iki ham sayısal kol da **0/8** yıl kazanır.
Sıfır shrinkage'lı yıllardaki makine hassasiyetinde MAE farkları bilimsel
kazanım sayılmaz; yıl toplamları frozen hesaplayıcının çıktısıdır.

Yıl içi paired moving-block bootstrap,10.000 tekrar,seed42:

| Karşılaştırma | Blok20 %95 kazanç aralığı | Blok60 %95 kazanç aralığı |
|---|---:|---:|
| numeric_D0–Naive |[−%0,2948; +%0,2274]|[−%0,2317; +%0,1793]|
| numeric_D1–Naive |[−%0,5720; +%0,0568]|[−%0,5248; +%0,0157]|
| numeric_D0–mask_D0 |[−%0,1149; +%0,2698]|[−%0,0887; +%0,2559]|
| numeric_D1–mask_D1 |[−%0,2423; +%0,0386]|[−%0,2205; +%0,0229]|

Texas'ın kontrol üzerine katkısı **belirsiz**; D0 nokta katkısı yaklaşık
%0,0804, D1 yaklaşık−%0,1019. Buna rağmen bu sabit seçilmiş tarifin **%5
pratik hedefi** her iki gecikme ve blokta dışarıda kalır. Bootstrap araştırma
geçmişindeki insan/model seçimlerini kapsayan bağımsız doğrulama değildir.

numeric_D0'nın yıllık Naive kazancı2016–2023:
`0,0,0,≈0,−%0,657,+0,+%0,191,−%0,142`.
Ağırlıklar `0,0,0,0,.25,0,.25,.25`; D1 `0,.25,0,0,.5,0,.25,.25`.
En büyük21 Naive hatası çıkarıldığında D0/D1 kazanç **−%0,0578/−%0,2955**.
Rapor yaşı0–4 alt kümesinde **+%0,0316/−%0,3955**; bunlar ön kayıtlı tanılardır,
yeni seçim kuralı veya başarılı alt grup iddiası değildir.

## Gözlenen mekanizma ve çıkarım sınırı

**STRONGLY SUPPORTED:** bu temsil/tarifte ek bilgi eğitim uyumunu artırırken
OOS tahmin hatasını azaltmıyor; geçmiş doğrulama tahminleri çoğu yılda sıfıra
yaklaştırıyor. numeric_D0 dış refitlerinin eğitim MAE kazancı yıllık ortalamada
%2,48'den %0,59'a inerken ham OOS kazanç−%3,76. Sayısal kollar ham olarak
kontrollerden de kötü. Küçültme zararı azaltıyor; pratik beceri üretmiyor.

Bu, bütün Texas bilgisinin işe yaramadığı veya daha fazla kapasitenin bunu
mutlaka çözeceği sonucunu vermez. D0'da1.080 origin'de kaynak yok;926 kondisyon
günü180 ayrı rapordan gelir. Planted151, squaring432, setting449, opening543,
harvested366 origin'de bilinir. Günlük satır sayısı bağımsız temel bilgi
örneği sayısı değildir. Koşullu etki, başka temsil ve gerçek vintage/saat
belirsizliği açık kalır; bunlar aynı grid'i büyütme gerekçesi değildir.

## Uygulama, doğrulama ve koruma

- Yeni adapter mevcut `Experiment/Ledger`, `inner_price` ve `predict_chunks`
  motorunu kullanır; ikinci eğitim motoru yok. T+5/yeni arama/release engellenir.
- Bilinmeyen rapor yaşı nullable metadata olarak yazılır; sonlu tahmin/hedef
  kontrolü gevşetilmedi. Hata sentetik testte, piyasa fit'lerinden önce bulundu.
- Fit bütçesi başarısız hesaplamaları sayar; checkpoint yayın hatası için
  doğrulanmış aynı fit tekrar eğitilmez. Sentetik fit ayrı ledger'dadır.
- 676 fit kimliği/payload hash'i, eğitim kohortu,120 warmup ve
  `target_date_5 < cutoff` denetlendi. Kaynak tarihlerinin varsayımsal erişimi
  karar sonrasına taşmıyor; bu tarihsel yayın saatinin doğruluğunu kanıtlamaz.
- Bağımsız yeniden çıkarım: **676 model**, maksimum log-getiri farkı
  `1,57161e-8`; **32** past-only shrinkage seçimi ve **24** güven aralığı
  sıfır yeni fit ile yeniden hesaplandı. Dört kolun hedef/tarihleri aynı.
- Tekrar `pilot` çağrısı32 çıktıyı cache'den doğruladı; fit sayıları676+1
  kaldı. İlk ve son piyasa makbuzu arası yaklaşık99 saniye; toplam kayıtlı
  hesaplama25,884 saniye. Oturum30 dakika sınırını aşmadı.
- **937 ML testi geçti,3 atlandı**,46 dar test/Ruff geçti. GPU çalıştırılmadı.
  Backend/frontend davranışı değiştirilmedi; GitHub CI ayrıca raporlanır.
- **949 özgün NASS girdisi**, önceki261 ön kayıt dosyası (mevcut cache dahil),
 40 eski kanıt dosyası,144 sicil nesnesi, eski trial nesneleri ve13 ileri
 kayıt dosyası korundu. Ana dirty checkout ve gece görevi değişmedi.

Yürütme kaynak kimliği `ca0d95b7cf585c4c3800d3bdb7be601dd12fb29256e2e33d1903e485ca42be71`
(233 dosya). Ön kayıt kimliği `aa4abe3e2b66d706cc38ed6cd53f91b3d194b5ac7212996ab8e311f3bdae24dd`.
Tahmin CSV SHA256 `0e68367e1425ac20323285525ddd37e7e87cb22f060942de39df52a629b2b530`.
3.945 üyeli ZIP SHA256 `44a7d13729dc497cc9a7fa98e80c1951548a6cba503d41ba6cdde93929ddab5a`.

## Komutlar ve sonraki tek karar

Önce `python ml/history.py check --query nass-regional --horizon 1`.
Yeni fit zaten denenmiş tam kapsam olarak sicile girdi. Eski sıfır-fit ön
kayıt tamamlanmış deneyle karıştırılmaz.

Yayın paketini ayrı klasöre açınca, mevcut kilitli CPU ortamında:

```powershell
python output/nass-regional-result-20261009/verify_result.py
python output/nass-regional-result-20261009/mechanism_check.py
```

Yeniden çıkarım eğitim yapmaz. Kaynak/environment kimliği eşleşmeyen güncel
checkout ile eski pilot devam ettirilmez; paketteki kaynak snapshot korunur.
CPU launcher profili `nass-regional-t1-pilot-v1` ve aşamaları
`prepare → pilot-plan → pilot → compare/report`; prepare için açık
`--registration-root` ve `--decision-contract`, ayrı `--drive-root` ve mevcut
`--cpu-environment` gerekir. Tamamlanan kapsam tekrar hazırlanıp eğitilmez.

**Karar:** sabit Texas/NASS fiyat programı büyütülmez, Naive korunur.
Sonraki somut operasyon kontrolü mevcut ileri görevde zamanında ilk yayın ve
kesim-öncesi makbuz zincirinin doğrulanmasıdır; gece takvimi değiştirilmez.
FAS'ın ilk-vintage hücresi dış kanıt bekliyor; aynı arama tekrarlanmaz.
Yeni piyasa fit'i için önce sonuçla elenmemiş, ayırıcı bir hipotez gerekir.
