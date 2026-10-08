# Bölgesel WASDE T+1 sonucu: sabit tarif pratik hedefi kurtarmıyor

**VERIFIED:** önceden kilitlenen dört kol, aynı 2019–2023 **1.254 origin** üzerinde
tamamlandı: **424 piyasa fit'i, 20 yıllık çıktı, 5.016 tahmin satırı**. Ayrı bir
sentetik Ridge kontrolü geçti; onun bir fit'i piyasa bütçesine veya performans
kanıtına karıştırılmadı. Yeni veri indirilmedi, eski fiyat/hedefler değiştirilmedi.

Önceden belirlenen karar: **`FIXED_RECIPE_BELOW_PRACTICAL_GOAL`**. Bu seçilmiş
tarif için %5 fiyat-MAE hedefi negatif kanıt aldı. Sayısal bilginin her olası
temsil/modelde yokluğu belirlenmedi. Naive korunur; aynı program büyük grid,
yeni model veya yeni kaynakla otomatik büyütülmez.

## Ana sonuç

Pozitif kazanç daha düşük MAE demektir; negatif değer zarar. Birimler kaynak
Cotton fiyatının birimleridir. Ortak Naive MAE **1,1837480962**.

| Kol | Seçilmiş fiyat-MAE | Naive kazancı | Ham Naive kazancı | Aktif oran | Aktif yönde başarı |
|---|---:|---:|---:|---:|---:|
| mask_D0 | 1,187324 | −%0,3021 | −%1,5476 | %20,02 | %50,20 |
| numeric_D0 | 1,189938 | **−%0,5229** | **−%10,2134** | %39,71 | %49,00 |
| mask_D1 | 1,187410 | −%0,3093 | −%1,5429 | %20,02 | %52,59 |
| numeric_D1 | 1,183875 | **−%0,0107** | **−%9,3435** | %19,62 | %49,19 |

Kontrolün %0,3'lük zararının azalması, Naive karşısında kullanılabilir üstünlük
değildir. D1 sayısal kolunun kontrol karşısında nokta katkısı var, aralık sıfırı
içeriyor. Birincil D0 katkısı da ayırt edilemiyor:

| Seçilmiş karşılaştırma | Blok20 %95 kazanç aralığı | Blok60 %95 kazanç aralığı |
|---|---:|---:|
| numeric_D0 / Naive | [−%1,1293; +%0,2785] | [−%1,2234; +%0,3093] |
| numeric_D1 / Naive | [−%0,1673; +%0,1490] | [−%0,1667; +%0,1488] |
| numeric_D0 / mask_D0 | [−%1,1947; +%0,7904] | [−%1,2127; +%0,8585] |
| numeric_D1 / mask_D1 | [−%0,4595; +%0,9529] | [−%0,4258; +%1,0691] |

10.000 tekrar, seed42, yıl sınırlarını koruyan mevcut paired bootstrap
kullanıldı. Her iki gecikmenin Naive kazancı üst sınırı, her iki blokta %5'in
altında. Aralıklar geçmişteki bütün insan/model seçimlerini kapsamaz; bağımsız
holdout veya ileri doğrulama değildir.

Ham sayısal kol, ham kontrole karşı da kötü: D0 kazanç aralıkları blok20/60
**[−%13,9485; −%0,9863] / [−%14,7228; −%1,0190]**; D1
**[−%12,8018; −%0,4953] / [−%13,4194; −%0,5021]**. Küçültme, bu ham zararların
çoğunu engelliyor; kaynakta sıfır sinyal olduğunun kanıtı değildir.

## Dönemler ve koşullu tanılar

| Yıl | numeric_D0 seçilmiş | numeric_D0 ham | numeric_D1 ham | D0/D1 seçilmiş ağırlık |
|---|---:|---:|---:|---:|
| 2019 | −%4,4214 | −%48,0405 | −%44,4901 | 0,25 / 0 |
| 2020 | %0 | −%15,7257 | −%12,2021 | 0 / 0 |
| 2021 | %0 | −%7,7739 | −%7,0383 | 0 / 0 |
| 2022 | %0 | −%0,5987 | −%0,8416 | 0 / 0 |
| 2023 | −%0,0159 | −%4,0251 | −%4,3660 | 0,25 / 0,25 |

Ham sayısal tahminler **0/5 yıl** Naive'ı yeniyor. En büyük %1 Naive hatası
çıkarıldığında ham kazanç D0 **−%10,8837**, D1 **−%10,0052**; zarar yalnız birkaç
uç gözlemden gelmiyor. Önceden tanımlanan kaynak yaşı 0–4 alt kümesinde de ham
MAE kazancı D0 **−%10,4554 (295 origin)**, D1 **−%10,3667 (236 origin)**.
Takvim etkisi veya iyi görünen alt küme sonradan yeni ana hedef yapılmadı.

Sıfır tahminler yön metriğinde “flat” sayılıyor. Bu yüzden seçilmiş tüm-origin
yön D0 %19,70, D1 %9,97; bunlar aktif yön başarısı değildir. Ham tüm-origin yön
%50,24/%50,48, aktif seçilmiş yön %49,00/%49,19. D0 yaş0–4 aktif yön %53,04
olsa da yalnız 115 aktif tahmin ve negatif MAE ile pratik üstünlük kanıtı değil.

## Gözlenen başarısızlık mekanizması

**VERIFIED:** sayısal alanlar eğitim uyumunu artırırken OOS tahmin genliğini ve
hatayı artırdı. 2019 dış refit'lerinin eğitim-MAE kazançlarının ortalaması
kontrolde %0,356, sayısal kolda %2,498; buna rağmen ham OOS kazanç −%4,84
civarındaki kontrol sonucundan **−%48,04**'e kötüleşiyor. Sayısal kolun ham
tahmin standart sapması **0,01734**, kontrolün **0,00328**; gerçekleşen günlük
getirinin standart sapması **0,01387**. Diğer dört yılda da sayısal tahmin
genliği kontrolün yaklaşık iki–üç katı, Naive MAE kazancı negatif.

Bu son paragraftaki eğitim/genlik analizi **sonradan yapılan açıklayıcı tanı**;
seçim veya birincil test değildir. Yeniden fit yapılmadı. Mevcut aday 95 rapor
taşıyor; OOS özelliklerde 60 ayrı rapor vintage'ı, en erken iç eğitimde yalnız
27 rapor var. Günlük tekrarlar 1.254 bağımsız temel bilgi yeniliği yaratmıyor.

**STRONGLY SUPPORTED:** bu temsilde ek kapasitenin ölçülebilir maliyeti var;
%5 düzeyinde kararlı T+1 katkısı elde edilmiyor. **PLAUSIBLE:** az sayıda aylık
yenilikten 26 seviye/revizyon ile günlük beklenen getiri öğrenmeye çalışmak,
geçmişe özgü ilişkileri büyütüyor. Bu deney, bilginin zaten Cotton fiyatında
olmasını, kötü temsilini ve daha küçük/koşullu bir etkiyi tek başına ayırmaz.
“Daha güçlü model gerekir” veya “WASDE işe yaramaz” sonucu çıkarılmaz.

## Bütünlük ve leakage sınırı

- Her kaynak değeri, [ön kayıttaki](WASDE_REGIONAL_T1_PREREGISTRATION_20261008.md)
  varsayımlı saat ve ortak eskilik sınırıyla seçildi. Kontrolün per-feature
  eksiklik bilgisi sayısal kolla aynı. Orijin/hedef/gerçekleşen fiyatlar eşit.
- 424 tamamlanmış fit'in eğitim satırları, beş gözlemlik olgunlaşması, checkpoint
  checksum'ları ve dış tahminleri yeniden doğrulandı. 20 küçültme seçimi yalnız
  geçmiş üç iç bloktan bağımsız yeniden hesaplandı.
- Median/mean/std ve hedef ölçeği her fit'in eğitiminden yeniden hesaplandı;
  kaydedilmiş Ridge katsayılarından çıkarım yeniden yapıldı. En büyük log-getiri
  farkı **1,49×10⁻⁸** (float32 hesap farkı); hiçbir model yeniden eğitilmedi.
- Fiyat-MAE, saklanan 5.016 tahmin satırından ayrı fiyat formülüyle eşleşti.
  Ayrı sentetik öğrenme kazancı **%99,7503**; eski TCN “passed” kaydı kullanılmadı.
- **2.682 eski dosya değişmedi.** İlk ön kayıt ve bütün eski sicil kayıtları
  korundu; sicile yalnız gerçekleşen piyasa deneyi ve sentetik kontrol eklendi.

**Tarihsel saat/vintage hâlâ doğrulanmış değil.** D0/D1 gecikme duyarlılığı
gerçek erişim kanıtı değildir; saat varsayımı yanlışsa gerçek zaman açısından
leakage mümkündür. Kaynak kabul kapısı, runtime modeli ve canlı tahmin değişmedi.
Beş görülmüş yıl korunmuş 6/8 koşulunu sağlayamaz; yayın kararı verilmedi.

ML yürütme kimliği:
`e7d172882aaac57fd0d42602a61bbd840863bb4e0d95a80bfd00026aed388436`.
Ön kayıt kimliği değişmedi:
`985c35f68fde70b7060a71852d08b907cc4a5f22b50660f81ea22167d4c68063`.
Yerel tüm ML kontrolü 704 geçti / 3 atlandı; son sicil koruması dahil 14 yürütme
testi ayrıca geçti. Ruff geçti. GPU test/eğitimi yok; backend/UI kodu değişmedi.

[Makine kanıtı](../research/evidence/wasde-regional-result-20261008.json),
[Release envanteri](../research/evidence/wasde-regional-result-release-20261008.json)
ve [ek kanıt paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-regional-result-20261008)
tam fit makbuzlarını, model/adapter'leri, tahminleri, frozen girdileri ve no-fit
`verify_result.py` / `mechanism_check.py` tekrar kontrollerini saklar.

Yürütme komutu, pinned CPU ortamı ve veri kökü açık verilerek:

```powershell
python ml/full_year_cpu.py pilot --profile wasde-regional-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root FROZEN_DATA_ROOT --max-minutes 30
python ml/full_year_cpu.py compare --profile wasde-regional-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root FROZEN_DATA_ROOT
```

Tamamlanan namespace üzerinde `pilot`, doğrulanmış cache'i okur; yeni fit
üretmez. Kod/ortam değişirse eski deneye devam edilmez. Yeni bir çalışma
kendiliğinden açılmaz: önce sicil ve bu sonucun kapsamı incelenir.
